import copy
import itertools

import pytest
import torch
import torch.nn.functional as F
from huggingface_hub import try_to_load_from_cache
from torch import nn

from model import LoRALinear, ViTFrameEncoder, ViTLoRAFrameEncoder
from model.vit.vit_lora_frame_encoder import canonical_targets, inject_lora


CHECKPOINT_ID = "google/vit-base-patch16-224"
PROJECTIONS = {"q": "q_proj", "k": "k_proj", "v": "v_proj"}
ALL_TARGETS = [list(combo) for size in (1, 2, 3) for combo in itertools.combinations("qkv", size)]


def tiny_pixels(batch_size=2, seed=1):
    return torch.randn(batch_size, 3, 32, 32, generator=torch.Generator().manual_seed(seed))


def lora_and_base(encoder):
    lora = dict(encoder.named_lora_parameters())
    base = {name: parameter for name, parameter in encoder.named_parameters() if name not in lora}
    return lora, base


def wrapped_paths(encoder):
    return [name for name, module in encoder.named_modules() if isinstance(module, LoRALinear)]


# --- target selection -------------------------------------------------------

def test_canonical_targets_orders_q_k_v():
    assert canonical_targets(["v", "q"]) == ("q", "v")
    assert canonical_targets(["k"]) == ("k",)
    assert canonical_targets(["v", "k", "q"]) == ("q", "k", "v")


@pytest.mark.parametrize("targets,error", [
    ([], ValueError), (["q", "q"], ValueError), (["x"], ValueError), (["Q"], ValueError),
    (["q", "o"], ValueError), ("qv", TypeError),
])
def test_invalid_target_modules(fake_pretrained_vit, targets, error):
    fake_pretrained_vit()
    with pytest.raises(error):
        ViTLoRAFrameEncoder("fake", target_modules=targets)


@pytest.mark.parametrize("kwargs", [{"rank": 0}, {"rank": -2}, {"alpha": 0}, {"dropout": 1.0}])
def test_invalid_lora_hyperparameters(fake_pretrained_vit, kwargs):
    fake_pretrained_vit()
    with pytest.raises(ValueError):
        ViTLoRAFrameEncoder("fake", **kwargs)


# --- structure, freezing, initialization -----------------------------------

@pytest.mark.parametrize("targets", ALL_TARGETS, ids=["".join(t) for t in ALL_TARGETS])
def test_all_target_combinations_wrap_only_selected_projections(fake_pretrained_vit, targets):
    source, calls = fake_pretrained_vit()
    encoder = ViTLoRAFrameEncoder("fake", target_modules=targets, rank=4, alpha=8)

    assert calls == [("fake", {"add_pooling_layer": False, "output_loading_info": True})]
    assert encoder.target_modules == tuple(t for t in "qkv" if t in targets)
    expected_paths = [f"vit.layers.{index}.attention.{PROJECTIONS[target]}"
                      for index in range(source.config.num_hidden_layers) for target in encoder.target_modules]
    assert sorted(wrapped_paths(encoder)) == sorted(expected_paths)
    assert sorted(f"vit.{path}" for path in encoder.lora_paths) == sorted(expected_paths)

    lora, base = lora_and_base(encoder)
    hidden = source.config.hidden_size
    assert len(lora) == 2 * len(expected_paths)
    assert sum(parameter.numel() for parameter in lora.values()) == len(expected_paths) * 4 * 2 * hidden
    assert all(parameter.requires_grad for parameter in lora.values())
    assert not any(parameter.requires_grad for parameter in base.values())
    assert {id(p) for p in encoder.parameters() if p.requires_grad} == {id(p) for p in lora.values()}

    for index, layer in enumerate(encoder.vit.layers):
        source_attention = source.layers[index].attention
        for target, name in PROJECTIONS.items():
            module = getattr(layer.attention, name)
            source_module = getattr(source_attention, name)
            if target in targets:
                assert isinstance(module, LoRALinear)
                assert torch.count_nonzero(module.lora_b.weight) == 0
                assert module.lora_a.weight.shape == (4, hidden)
                assert module.lora_b.weight.shape == (hidden, 4)
                module = module.base
            assert type(module) is nn.Linear
            assert torch.equal(module.weight, source_module.weight)
            assert torch.equal(module.bias, source_module.bias)
        assert type(layer.attention.o_proj) is nn.Linear

    features = encoder(tiny_pixels())
    assert features.shape == (2, hidden)
    assert torch.isfinite(features).all()


def test_disabled_builds_no_lora_and_is_frozen_cls_encoder(fake_pretrained_vit):
    source, _ = fake_pretrained_vit()
    encoder = ViTLoRAFrameEncoder("fake", enabled=False, target_modules=[]).eval()
    assert encoder.target_modules == () and encoder.lora_paths == []
    assert wrapped_paths(encoder) == []
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    pixels = tiny_pixels()
    with torch.no_grad():
        torch.testing.assert_close(encoder(pixels), source(pixel_values=pixels).last_hidden_state[:, 0, :])


@pytest.mark.parametrize("targets", ALL_TARGETS, ids=["".join(t) for t in ALL_TARGETS])
def test_zero_b_matches_lora_off_and_nonzero_b_changes_cls(fake_pretrained_vit, targets):
    fake_pretrained_vit()  # every encoder below is built from the same ViT weights
    lora_off = ViTLoRAFrameEncoder("fake", enabled=False).eval()
    lora_on = ViTLoRAFrameEncoder("fake", target_modules=targets).eval()
    pixels = tiny_pixels()
    with torch.no_grad():
        torch.testing.assert_close(lora_on(pixels), lora_off(pixels))
        for name, parameter in lora_on.named_lora_parameters():
            if ".lora_b." in name:
                parameter.normal_()
        assert not torch.allclose(lora_on(pixels), lora_off(pixels))


def test_loading_report_with_missing_keys_fails(fake_pretrained_vit):
    fake_pretrained_vit(loading_info={"missing_keys": ["layers.0.attention.q_proj.weight"]})
    with pytest.raises(RuntimeError, match="missing_keys"):
        ViTLoRAFrameEncoder("fake")


# --- fail-fast injection ----------------------------------------------------

def test_inject_keeps_original_linear_objects(fake_pretrained_vit):
    vit = copy.deepcopy(fake_pretrained_vit()[0])
    originals = [layer.attention.q_proj for layer in vit.layers]
    paths = inject_lora(vit, ("q",), rank=2, alpha=2, dropout=0.0)
    assert paths == [f"layers.{index}.attention.q_proj" for index in range(len(vit.layers))]
    assert all(layer.attention.q_proj.base is original for layer, original in zip(vit.layers, originals))


def test_double_injection_fails_without_changes(fake_pretrained_vit):
    vit = copy.deepcopy(fake_pretrained_vit()[0])
    inject_lora(vit, ("q", "v"), rank=2, alpha=2, dropout=0.0)
    before = [name for name, module in vit.named_modules() if isinstance(module, LoRALinear)]
    with pytest.raises(RuntimeError, match="already injected"):
        inject_lora(vit, ("k", "v"), rank=2, alpha=2, dropout=0.0)
    assert [name for name, module in vit.named_modules() if isinstance(module, LoRALinear)] == before


@pytest.mark.parametrize("corruption,error", [
    ("missing", TypeError), ("non_linear", TypeError), ("shape", RuntimeError), ("layer_count", RuntimeError),
])
def test_unexpected_structure_fails_before_any_injection(fake_pretrained_vit, corruption, error):
    vit = copy.deepcopy(fake_pretrained_vit()[0])
    if corruption == "missing":
        del vit.layers[1].attention.k_proj
    elif corruption == "non_linear":
        vit.layers[1].attention.k_proj = nn.Identity()
    elif corruption == "shape":
        vit.layers[1].attention.k_proj = nn.Linear(vit.config.hidden_size, 8)
    else:
        del vit.layers[1]
    with pytest.raises(error):
        inject_lora(vit, ("q", "k"), rank=2, alpha=2, dropout=0.0)
    assert not any(isinstance(module, LoRALinear) for module in vit.modules())


# --- gradients and optimizer updates (engineering-only loss, no head) -------

def assert_two_step_lora_only_updates(encoder, pixels, target):
    """Engineering-only MSE(CLS, fixed target) loss; optimizer holds LoRA A/B only."""
    encoder.train()
    lora, base = lora_and_base(encoder)
    a_names = [name for name in lora if ".lora_a." in name]
    b_names = [name for name in lora if ".lora_b." in name]
    base_before = {name: parameter.detach().clone() for name, parameter in base.items()}
    optimizer = torch.optim.AdamW(lora.values(), lr=1e-3, weight_decay=0.0)
    assert ({id(p) for group in optimizer.param_groups for p in group["params"]}
            == {id(p) for p in encoder.parameters() if p.requires_grad})

    def step():
        optimizer.zero_grad(set_to_none=True)
        features = encoder(pixels)
        assert features.requires_grad
        loss = F.mse_loss(features, target)
        loss.backward()
        assert torch.isfinite(loss)
        grads = {name: parameter.grad.detach().clone() for name, parameter in lora.items()}
        before = {name: parameter.detach().clone() for name, parameter in lora.items()}
        optimizer.step()
        assert all(torch.isfinite(grad).all() for grad in grads.values())
        assert all(torch.isfinite(parameter).all() for parameter in lora.values())
        assert all(parameter.grad is None for parameter in base.values())
        changed = {name for name, parameter in lora.items() if not torch.equal(parameter, before[name])}
        return grads, changed

    grads, changed = step()
    # B == 0 at init, so dL/dA == 0 on the first step by design; B gets a gradient.
    assert all(torch.count_nonzero(grads[name]) == 0 for name in a_names)
    assert any(torch.count_nonzero(grads[name]) > 0 for name in b_names)
    assert changed and changed <= set(b_names)

    grads, changed = step()
    assert any(torch.count_nonzero(grads[name]) > 0 for name in a_names)
    assert changed & set(a_names)

    assert all(torch.equal(parameter, base_before[name]) for name, parameter in base.items())
    assert not any(parameter.requires_grad for parameter in base.values())


def test_lora_only_gradients_and_adamw_updates(fake_pretrained_vit):
    source, _ = fake_pretrained_vit()
    encoder = ViTLoRAFrameEncoder("fake", target_modules=["q", "v"])
    target = torch.randn(4, source.config.hidden_size, generator=torch.Generator().manual_seed(2))
    assert_two_step_lora_only_updates(encoder, tiny_pixels(batch_size=4), target)


# --- save / reload ----------------------------------------------------------

def test_lora_state_dict_round_trip(fake_pretrained_vit):
    fake_pretrained_vit()
    trained = ViTLoRAFrameEncoder("fake", target_modules=["q", "k", "v"]).eval()
    with torch.no_grad():
        for _, parameter in trained.named_lora_parameters():
            parameter.normal_()
    state = trained.lora_state_dict()
    assert set(state) == {name for name, _ in trained.named_lora_parameters()}

    reloaded = ViTLoRAFrameEncoder("fake", target_modules=["q", "k", "v"]).eval()
    pixels = tiny_pixels()
    with torch.no_grad():
        assert not torch.allclose(reloaded(pixels), trained(pixels))
        reloaded.load_lora_state_dict(state)
        torch.testing.assert_close(reloaded(pixels), trained(pixels))


@pytest.mark.parametrize("problem", ["missing", "base_key", "other_targets"])
def test_load_lora_state_dict_rejects_mismatched_keys(fake_pretrained_vit, problem):
    source, _ = fake_pretrained_vit()
    encoder = ViTLoRAFrameEncoder("fake", target_modules=["q", "v"])
    state = encoder.lora_state_dict()
    if problem == "missing":
        state.pop(next(iter(state)))
    elif problem == "base_key":
        state["vit.layernorm.weight"] = torch.zeros(source.config.hidden_size)
    else:
        state = ViTLoRAFrameEncoder("fake", target_modules=["k"]).lora_state_dict()
    with pytest.raises(KeyError):
        encoder.load_lora_state_dict(state)


# --- real google/vit-base-patch16-224 (local Hugging Face cache only) -------

@pytest.fixture(scope="module")
def vit_base_off():
    if not isinstance(try_to_load_from_cache(CHECKPOINT_ID, "config.json"), str):
        pytest.skip(f"{CHECKPOINT_ID} is not in the local Hugging Face cache")
    return ViTLoRAFrameEncoder(CHECKPOINT_ID, enabled=False).eval()


def vit_base_pixels(batch_size=2):
    return torch.rand(batch_size, 3, 224, 224, generator=torch.Generator().manual_seed(3)) * 2 - 1


def test_vit_base_default_qv_loading_report_and_counts(vit_base_off):
    encoder = ViTLoRAFrameEncoder(CHECKPOINT_ID)
    assert not encoder.loading_info["missing_keys"]
    assert not encoder.loading_info["mismatched_keys"]
    assert sorted(encoder.loading_info["unexpected_keys"]) == ["classifier.bias", "classifier.weight"]
    assert encoder.vit.pooler is None
    assert encoder.target_modules == ("q", "v")
    assert len(wrapped_paths(encoder)) == 24
    lora, base = lora_and_base(encoder)
    assert len(lora) == 48
    assert sum(parameter.numel() for parameter in lora.values()) == 294912
    assert sum(p.numel() for p in encoder.parameters() if p.requires_grad) == 294912
    assert not any(parameter.requires_grad for parameter in base.values())


@pytest.mark.parametrize("targets", ALL_TARGETS, ids=["".join(t) for t in ALL_TARGETS])
def test_vit_base_all_combinations_counts_and_initial_equivalence(vit_base_off, monkeypatch, targets):
    monkeypatch.setattr(
        "model.vit.vit_lora_frame_encoder.ViTModel.from_pretrained",
        lambda checkpoint_id, **kwargs: (copy.deepcopy(vit_base_off.vit), copy.deepcopy(vit_base_off.loading_info)))
    encoder = ViTLoRAFrameEncoder(CHECKPOINT_ID, target_modules=targets).eval()
    assert len(wrapped_paths(encoder)) == 12 * len(targets)
    lora, _ = lora_and_base(encoder)
    assert len(lora) == 24 * len(targets)
    assert sum(parameter.numel() for parameter in lora.values()) == 147456 * len(targets)
    pixels = vit_base_pixels(batch_size=1)
    with torch.no_grad():
        features = encoder(pixels)
        assert features.shape == (1, 768)
        torch.testing.assert_close(features, vit_base_off(pixels))


def test_vit_base_lora_off_matches_existing_frame_encoder(vit_base_off):
    existing = ViTFrameEncoder(CHECKPOINT_ID).eval()
    pixels = vit_base_pixels()
    with torch.no_grad():
        expected = existing(pixels)
        actual = vit_base_off(pixels)
    assert actual.shape == expected.shape == (2, 768)
    torch.testing.assert_close(actual, expected)


def test_vit_base_qv_two_step_lora_only_updates(vit_base_off, monkeypatch):
    monkeypatch.setattr(
        "model.vit.vit_lora_frame_encoder.ViTModel.from_pretrained",
        lambda checkpoint_id, **kwargs: (copy.deepcopy(vit_base_off.vit), copy.deepcopy(vit_base_off.loading_info)))
    torch.manual_seed(0)
    encoder = ViTLoRAFrameEncoder(CHECKPOINT_ID, target_modules=["q", "v"])
    target = torch.randn(2, 768, generator=torch.Generator().manual_seed(2))
    assert_two_step_lora_only_updates(encoder, vit_base_pixels(), target)
