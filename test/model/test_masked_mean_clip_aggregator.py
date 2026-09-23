import pytest
import torch

from model import MaskedMeanClipAggregator


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_all_valid_matches_mean_and_preserves_shape_dtype_device(dtype):
    features = torch.arange(24, dtype=dtype).reshape(4, 6)
    aggregator = MaskedMeanClipAggregator()
    output = aggregator(features, torch.ones(4, dtype=torch.bool))
    torch.testing.assert_close(output, features.mean(dim=0))
    assert output.shape == (6,)
    assert output.dtype == features.dtype
    assert output.device == features.device
    assert list(aggregator.parameters()) == []


@pytest.mark.parametrize('padding_value', [0.0, 1e30, float('nan'), float('inf')])
def test_invalid_rows_do_not_contribute_values_or_denominator(padding_value):
    features = torch.tensor([[2., 4.], [padding_value, padding_value], [6., 12.]])
    output = MaskedMeanClipAggregator()(features, torch.tensor([True, False, True]))
    torch.testing.assert_close(output, torch.tensor([4., 8.]))


@pytest.mark.parametrize('order', [[3, 2, 1, 0], [2, 0, 3, 1]])
def test_reverse_and_permutation_preserve_output(order):
    features = torch.tensor([[1., 8.], [1e20, 1e20], [4., 2.], [7., 5.]])
    mask = torch.tensor([True, False, True, True])
    aggregator = MaskedMeanClipAggregator()
    torch.testing.assert_close(aggregator(features[order], mask[order]), aggregator(features, mask))


@pytest.mark.parametrize('features,mask,message', [
    (torch.zeros(2, 3, 4), torch.ones(2, dtype=torch.bool), 'frame_features'),
    (torch.zeros(2, 3), torch.ones(2, 1, dtype=torch.bool), 'valid_mask'),
    (torch.zeros(2, 3), torch.ones(3, dtype=torch.bool), 'same T'),
    (torch.zeros(2, 3), torch.zeros(2, dtype=torch.bool), 'at least one'),
    (torch.zeros(0, 3), torch.zeros(0, dtype=torch.bool), 'at least one'),
])
def test_invalid_shapes_and_empty_mask_raise(features, mask, message):
    with pytest.raises(ValueError, match=message):
        MaskedMeanClipAggregator()(features, mask)


def test_non_boolean_mask_is_rejected_instead_of_integer_indexing():
    with pytest.raises(TypeError, match='torch.bool'):
        MaskedMeanClipAggregator()(torch.zeros(2, 3), torch.tensor([1, 0]))


def test_backward_only_reaches_valid_rows_with_correct_scaling():
    features = torch.arange(12, dtype=torch.float64).reshape(4, 3).requires_grad_()
    mask = torch.tensor([True, False, True, False])
    output = MaskedMeanClipAggregator()(features, mask)
    output.sum().backward()
    torch.testing.assert_close(features.grad, torch.tensor([
        [0.5, 0.5, 0.5], [0., 0., 0.], [0.5, 0.5, 0.5], [0., 0., 0.],
    ], dtype=torch.float64))
