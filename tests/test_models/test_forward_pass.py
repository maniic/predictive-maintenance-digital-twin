"""Tests for model forward pass shapes."""

import pytest
import torch

from src.models.cnn import TemporalCNNModel
from src.models.lstm import LSTMModel
from src.models.transformer import TransformerModel

BATCH_SIZE = 4
SEQ_LEN = 30
INPUT_DIM = 14


@pytest.fixture
def sample_input():
    return torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)


class TestLSTMForwardPass:
    def test_output_shape(self, sample_input):
        model = LSTMModel(
            input_dim=INPUT_DIM, sequence_length=SEQ_LEN, hidden_size=32, num_layers=1
        )
        model.eval()
        with torch.no_grad():
            output = model(sample_input)
        assert output.shape == (BATCH_SIZE,)

    def test_output_is_finite(self, sample_input):
        model = LSTMModel(
            input_dim=INPUT_DIM, sequence_length=SEQ_LEN, hidden_size=32, num_layers=1
        )
        model.eval()
        with torch.no_grad():
            output = model(sample_input)
        assert torch.isfinite(output).all()


class TestCNNForwardPass:
    def test_output_shape(self, sample_input):
        model = TemporalCNNModel(input_dim=INPUT_DIM, sequence_length=SEQ_LEN, channels=[16, 32])
        model.eval()
        with torch.no_grad():
            output = model(sample_input)
        assert output.shape == (BATCH_SIZE,)

    def test_output_is_finite(self, sample_input):
        model = TemporalCNNModel(input_dim=INPUT_DIM, sequence_length=SEQ_LEN, channels=[16, 32])
        model.eval()
        with torch.no_grad():
            output = model(sample_input)
        assert torch.isfinite(output).all()


class TestTransformerForwardPass:
    def test_output_shape(self, sample_input):
        model = TransformerModel(
            input_dim=INPUT_DIM, sequence_length=SEQ_LEN, d_model=32, n_heads=4, n_layers=1
        )
        model.eval()
        with torch.no_grad():
            output = model(sample_input)
        assert output.shape == (BATCH_SIZE,)

    def test_output_is_finite(self, sample_input):
        model = TransformerModel(
            input_dim=INPUT_DIM, sequence_length=SEQ_LEN, d_model=32, n_heads=4, n_layers=1
        )
        model.eval()
        with torch.no_grad():
            output = model(sample_input)
        assert torch.isfinite(output).all()
