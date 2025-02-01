import unittest

import torch

from resnet_10.configuration_resnet import ResNet10Config
from resnet_10.modeling_resnet import ResNet10


class TestResNet10(unittest.TestCase):
    def setUp(self):
        self.config = ResNet10Config(
            num_channels=3,
            embedding_size=64,
            hidden_sizes=[64, 128, 256, 512],
            hidden_act="relu",
            output_hidden_states=False,
        )
        self.model = ResNet10(self.config)
        self.batch_size = 2
        self.input_size = 224

    def test_model_initialization(self):
        """Test if model initializes correctly with given config"""
        self.assertIsInstance(self.model, ResNet10)
        self.assertEqual(self.model.config, self.config)

    def test_forward_pass_shape(self):
        """Test if forward pass produces correct output shape"""
        x = torch.randn(self.batch_size, 3, self.input_size, self.input_size)
        output = self.model(x)

        # Calculate expected output size
        # After initial conv and maxpool: 224 -> 112 -> 56
        # After 4 stages with stride=2: 56 -> 28 -> 14 -> 7
        expected_size = 7

        self.assertEqual(output.last_hidden_state.shape, (self.batch_size, 512, expected_size, expected_size))

    def test_hidden_states_output(self):
        """Test if hidden states are returned when requested"""
        x = torch.randn(self.batch_size, 3, self.input_size, self.input_size)

        # Test with output_hidden_states=True
        output = self.model(x, output_hidden_states=True)

        # Should have hidden states from:
        # 1. After embedder (before encoder)
        # 2. After each encoder stage (4 stages)
        # Total: 5 hidden states
        self.assertIsNotNone(output.hidden_states)
        self.assertEqual(len(output.hidden_states), 5)

    def test_different_batch_sizes(self):
        """Test if model handles different batch sizes"""
        batch_sizes = [1, 4, 8]
        for bs in batch_sizes:
            x = torch.randn(bs, 3, self.input_size, self.input_size)
            output = self.model(x)
            self.assertEqual(output.last_hidden_state.shape[0], bs)

    def test_model_parameters_not_none(self):
        """Test if all model parameters are initialized (not None)"""
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param)
            self.assertIsNotNone(param.data)
            self.assertFalse(torch.isnan(param.data).any())

    def test_input_channel_validation(self):
        """Test if model properly handles incorrect number of input channels"""
        wrong_channels = 4  # Config expects 3
        x = torch.randn(self.batch_size, wrong_channels, self.input_size, self.input_size)

        with self.assertRaises(RuntimeError):
            _ = self.model(x)


if __name__ == "__main__":
    unittest.main()
