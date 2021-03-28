import torch
from torch import nn
from math import pi


class Envelope(nn.Module):

    def __init__(self, exponent, ):
        super(Envelope, self).__init__()
        self.exponent = exponent

        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, inputs):

        # Envelope function divided by r
        env_val = 1 / inputs + self.a * inputs ** (self.p - 1) + self.b * inputs ** self.p + self.c * inputs ** (
                    self.p + 1)
        return torch.where(inputs < 1, env_val, torch.zeros_like(inputs))

    def forward(self, inputs):
        d_scaled = (inputs) * self.inv_cutoff
        # Necessary for proper broadcasting behaviour
        d_scaled = d_scaled.unsqueeze(-1)
        # d_cutoff = self.envelope(d_scaled)
        if self.use_cutoff:
            d_cutoff = torch.cos(d_scaled)
            d_cutoff *= (d_scaled <= 1).float()
        else:
            d_cutoff = torch.tensor(1.)
        if self.with_cos:
            out_sin = torch.sin(self.frequencies * d_scaled)
            out_cos = torch.cos(self.frequencies * d_scaled)
            out = d_cutoff * torch.cat([out_sin, out_cos], dim=-1)
        else:
            out = d_cutoff * torch.sin(self.frequencies * d_scaled)

        if self.linear_mapping is not None:
            out = self.linear_mapping(out)

        return out


class TransformerPositionalEncoding(nn.Module):

    def __init__(self, d_model, pos_emb_base_pow=10000, **kwargs):
        super(TransformerPositionalEncoding, self).__init__()
        self.pos_emb_base_pow = pos_emb_base_pow
        self.d_model = d_model

    def forward(self, distance_bins):
        device = distance_bins.device
        freq_seq = torch.arange(0, self.d_model, 2.0, dtype=torch.float, device=device)
        inv_freq = 1 / torch.pow(self.pos_emb_base_pow, (freq_seq / self.d_model))

        batch_size, num_bins = distance_bins.shape
        dists_flat = distance_bins.reshape(-1)

        sinusoid_inp = torch.einsum('i,d->id', dists_flat, inv_freq)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb.reshape([batch_size, num_bins, self.d_model])

        return pos_emb