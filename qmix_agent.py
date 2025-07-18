import torch
import torch.nn as nn
from offpolicy.utils.util import to_torch
from offpolicy.algorithms.utils.rnn import RNNBase
from offpolicy.algorithms.utils.act import ACTLayer
from types import SimpleNamespace

class Qmix(nn.Module):
    def __init__(self):
        super(Qmix, self).__init__()

        self.args = SimpleNamespace()
        self.args.input_dim = 18
        self.args.act_dim = 5
        self.args.recurrent_N = 1
        self.args.use_feature_normalization = True
        self.args.use_orthogonal = True
        self.args.use_ReLU = True
        self.args.use_conv1d = False
        self.args.stacked_frames = 1
        self.args.layer_N = 1
        self.args.hidden_size = 64
        self.args.gain = 0.01
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.args.tpdv = dict(dtype = torch.float32, device = self.args.device)

        self.rnn = RNNBase(self.args, self.args.input_dim)
        self.q = ACTLayer(self.args.act_dim, self.args.hidden_size, self.args.use_orthogonal, gain = self.args.gain)

        self.load_state_dict(torch.load("./models/trained_qmix.pt"))

    @torch.no_grad
    def forward(self, obs, rnn_states):
        obs = to_torch(obs).to(**self.args.tpdv)
        obs = obs[None]
        rnn_states = to_torch(rnn_states).to(**self.args.tpdv)
        rnn_states = rnn_states[None]
        rnn_outs, h_final = self.rnn(obs, rnn_states)
        q_outs = self.q(rnn_outs, True)
        return q_outs, h_final