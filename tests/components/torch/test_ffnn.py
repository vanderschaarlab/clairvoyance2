import torch.nn as nn

from clairvoyance2.components.torch.ffnn import FeedForwardNet


class TestFeedForwardNN:
    def test_no_hidden_dims(self):
        ffnn = FeedForwardNet(in_dim=5, out_dim=1)

        assert len(ffnn.seq) == 2
        assert isinstance(ffnn.seq[0], nn.Linear)
        assert ffnn.seq[0].in_features == 5
        assert ffnn.seq[0].out_features == 1
        assert isinstance(ffnn.seq[1], nn.ReLU)

    def test_single_hidden_dim(self):
        ffnn = FeedForwardNet(in_dim=5, out_dim=1, hidden_dims=[3])

        assert len(ffnn.seq) == 4
        assert isinstance(ffnn.seq[0], nn.Linear)
        assert ffnn.seq[0].in_features == 5
        assert ffnn.seq[0].out_features == 3
        assert isinstance(ffnn.seq[1], nn.ReLU)
        assert isinstance(ffnn.seq[2], nn.Linear)
        assert ffnn.seq[2].in_features == 3
        assert ffnn.seq[2].out_features == 1
        assert isinstance(ffnn.seq[3], nn.ReLU)

    def test_multiple_hidden_dim(self):
        ffnn = FeedForwardNet(in_dim=5, out_dim=1, hidden_dims=[3, 2])

        assert len(ffnn.seq) == 6
        assert isinstance(ffnn.seq[0], nn.Linear)
        assert ffnn.seq[0].in_features == 5
        assert ffnn.seq[0].out_features == 3
        assert isinstance(ffnn.seq[1], nn.ReLU)
        assert isinstance(ffnn.seq[2], nn.Linear)
        assert ffnn.seq[2].in_features == 3
        assert ffnn.seq[2].out_features == 2
        assert isinstance(ffnn.seq[3], nn.ReLU)
        assert isinstance(ffnn.seq[4], nn.Linear)
        assert ffnn.seq[4].in_features == 2
        assert ffnn.seq[4].out_features == 1
        assert isinstance(ffnn.seq[5], nn.ReLU)

    def test_set_activations(self):
        ffnn = FeedForwardNet(in_dim=5, out_dim=1, hidden_dims=[3], out_activation="Sigmoid", hidden_activations="Tanh")

        assert len(ffnn.seq) == 4
        assert isinstance(ffnn.seq[1], nn.Tanh)
        assert isinstance(ffnn.seq[3], nn.Sigmoid)

    def test_set_activations_None(self):
        ffnn = FeedForwardNet(in_dim=5, out_dim=1, hidden_dims=[3], out_activation=None, hidden_activations=None)

        assert len(ffnn.seq) == 2
        assert isinstance(ffnn.seq[0], nn.Linear)
        assert ffnn.seq[0].in_features == 5
        assert ffnn.seq[0].out_features == 3
        assert isinstance(ffnn.seq[1], nn.Linear)
        assert ffnn.seq[1].in_features == 3
        assert ffnn.seq[1].out_features == 1

    def test_set_out_activation_softmax(self):
        ffnn = FeedForwardNet(in_dim=5, out_dim=1, hidden_dims=[3], out_activation="Softmax")

        assert isinstance(ffnn.seq[-1], nn.Softmax)
        assert ffnn.seq[-1].dim == -1
