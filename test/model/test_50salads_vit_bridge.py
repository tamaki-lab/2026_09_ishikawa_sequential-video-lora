import unittest

import sequential_loader as sl
import torch

from integration.sequential_vit import encode_chunk


class RecordingProcessor:
    def __call__(self, *, images, return_tensors):
        assert return_tensors == "pt"
        self.images = images.clone()
        pixel_values = images[:, :, :1, :1].float().expand(-1, 3, 224, 224).contiguous()
        return {"pixel_values": pixel_values}


class ValueEncoder:
    def __call__(self, pixel_values):
        return pixel_values[:, 0, 0, 0, None].expand(-1, 768).contiguous()


class Test50SaladsViTBridge(unittest.TestCase):
    def test_padding_is_skipped_and_features_keep_frame_positions(self):
        frames = torch.zeros((16, 3, 2, 2), dtype=torch.uint8)
        frames[0].fill_(3)
        frames[1].fill_(7)
        frames[2].fill_(200)
        sample = sl.SequentialSample(
            frames=frames,
            frame_indices=torch.tensor([0, 1] + [-1] * 14, dtype=torch.int64),
            timestamps=torch.tensor([1.0, 0.0] + [float("nan")] * 14, dtype=torch.float64),
            valid_mask=torch.tensor([True, True] + [False] * 14),
            sequence_id="01-1",
            sequence_index=0,
            is_first=True,
            is_last=True,
            source_id="01-1",
            sequence_length=1,
            evaluation_reference=object(),
        )
        original_indices = sample.frame_indices.clone()
        original_timestamps = sample.timestamps.clone()
        processor = RecordingProcessor()

        with torch.no_grad():
            pixel_values, valid_features, frame_features = encode_chunk(
                sample, processor, ValueEncoder(), torch.device("cpu")
            )

        self.assertEqual(tuple(processor.images.shape), (2, 3, 2, 2))
        self.assertEqual(processor.images[:, 0, 0, 0].tolist(), [3, 7])
        self.assertEqual(tuple(pixel_values.shape), (2, 3, 224, 224))
        self.assertEqual(tuple(valid_features.shape), (2, 768))
        self.assertEqual(tuple(frame_features.shape), (16, 768))
        self.assertEqual(frame_features[:, 0].tolist(), [3.0, 7.0] + [0.0] * 14)
        self.assertTrue(torch.all(frame_features[2:] == 0).item())
        self.assertTrue(torch.equal(sample.frame_indices, original_indices))
        self.assertTrue(torch.allclose(sample.timestamps, original_timestamps, equal_nan=True))


if __name__ == "__main__":
    unittest.main()
