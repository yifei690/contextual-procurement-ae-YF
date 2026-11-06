"""Predictor registry: common inputs/heads, with explicit gradient capabilities."""
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


@dataclass
class Forecast:
    location: torch.Tensor
    scale: torch.Tensor
    embedding: torch.Tensor
    mask: torch.Tensor


class CommonHeads(nn.Module):
    def __init__(self, width, horizon, fields, embedding=32):
        super().__init__()
        self.horizon, self.fields = horizon, fields
        self.location = nn.Linear(width, horizon*fields)
        self.scale = nn.Linear(width, horizon*fields)
        self.embedding = nn.Linear(width, embedding)
        nn.init.zeros_(self.location.weight)
        nn.init.zeros_(self.location.bias)
        nn.init.zeros_(self.scale.weight)
        nn.init.constant_(self.scale.bias, -1.)

    def forward(self, encoded, mask):
        shape = (-1, self.horizon, self.fields)
        return Forecast(self.location(encoded).reshape(shape),
                        .05+2.95*torch.sigmoid(self.scale(encoded)).reshape(shape),
                        self.embedding(encoded), mask)


class CausalBlock(nn.Module):
    def __init__(self, incoming, channels, dilation):
        super().__init__()
        self.pad = 2*dilation
        self.conv1 = nn.Conv1d(incoming, channels, 3, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.skip = nn.Conv1d(incoming, channels, 1) if incoming != channels else nn.Identity()

    def forward(self, x):
        z = F.silu(self.conv1(F.pad(x, (self.pad, 0))))
        z = F.silu(self.conv2(F.pad(z, (self.pad, 0))))
        return F.silu(z+self.skip(x))


class NeuralPredictor(nn.Module):
    supports_jpo = True
    def __init__(self, architecture, history=168, input_fields=5, future_fields=6, horizon=25,
                 output_fields=5, embedding=32):
        super().__init__()
        self.architecture = architecture
        self.history, self.horizon = history, horizon
        self.future_encoder = nn.Sequential(nn.Linear(horizon*future_fields, 32), nn.Tanh())
        if architecture == "bp":
            self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(history*input_fields, 128), nn.Tanh(),
                                         nn.Linear(128, 64), nn.Tanh())
        elif architecture == "lstm":
            self.encoder = nn.LSTM(input_fields, 64, batch_first=True)
        elif architecture == "tcn":
            self.encoder = nn.Sequential(*[CausalBlock(input_fields if i == 0 else 32, 32, 2**i) for i in range(7)])
        elif architecture == "transformer":
            self.input_projection = nn.Linear(input_fields, 64)
            layer = nn.TransformerEncoderLayer(64, 4, dim_feedforward=128, dropout=0., activation="gelu", batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            position = torch.arange(history)[:, None]
            rates = torch.exp(torch.arange(0, 64, 2)*(-math.log(10000.)/64))
            pe = torch.zeros(history, 64)
            pe[:, 0::2], pe[:, 1::2] = torch.sin(position*rates), torch.cos(position*rates)
            self.register_buffer("positional", pe)
        else:
            raise ValueError(f"unknown neural architecture: {architecture}")
        width = 32 if architecture == "tcn" else 64
        self.fusion = nn.Sequential(nn.Linear(width+32, 64), nn.Tanh())
        self.heads = CommonHeads(64, horizon, output_fields, embedding)

    def forward(self, history, future, mask):
        if history.shape[1] != self.history or future.shape[1] != self.horizon:
            raise ValueError("history/horizon mismatch")
        if self.architecture == "lstm":
            _, (hidden, _) = self.encoder(history)
            encoded = hidden[-1]
        elif self.architecture == "tcn":
            encoded = self.encoder(history.transpose(1, 2))[:, :, -1]
        elif self.architecture == "transformer":
            encoded = self.encoder(self.input_projection(history)+self.positional[None])[:, -1]
        else:
            encoded = self.encoder(history)
        known = self.future_encoder((future*mask[..., None]).flatten(1))
        return self.heads(self.fusion(torch.cat([encoded, known], -1)), mask)

    def configure_training(self, scope):
        if scope not in ("full", "heads"):
            raise ValueError("training scope must be full or heads")
        for parameter in self.parameters():
            parameter.requires_grad_(scope == "full")
        for parameter in self.heads.parameters():
            parameter.requires_grad_(True)


class TraditionalPredictor:
    """Direct multi-horizon seasonal AR or boosted trees; scales are calibrated.

    Ridge uses the same available lagged windows and calendar features as the
    neural models. Its fixed regularization is shared across all output models.
    HistGBR is fitted separately by variable with delivery calendar and recent
    lag summaries; this adapter is forward-only and does not claim native
    probabilistic or neural embeddings.
    """
    supports_jpo = False
    def __init__(self, architecture, horizon=25, output_fields=5, seed=0):
        self.architecture, self.horizon, self.fields, self.seed = architecture, horizon, output_fields, seed
        self.models = []
        self.scale = np.ones(output_fields)

    def _design(self, history, future):
        h = np.asarray(history)
        # Aligned common windows; no contemporaneous target values.
        selected = h[:, [-1, -2, -24, -25, -48, -72, -168], :].reshape(len(h), -1)
        summary = np.concatenate([selected, h.mean(1), h.std(1)], 1)
        if self.architecture == "ar":
            return np.concatenate([h.reshape(len(h), -1), future.reshape(len(h), -1)], 1)
        return np.concatenate([np.repeat(summary[:, None, :], self.horizon, 1), future], -1)

    def fit(self, history, future, target, mask):
        x = self._design(history, future)
        if self.architecture == "ar":
            # Masked horizon-wise regression avoids fitting artificial padded zeros.
            for t in range(self.horizon):
                ok = np.asarray(mask)[:, t].astype(bool)
                if ok.sum() < 2:
                    self.models.append(None)
                else:
                    self.models.append(Ridge(alpha=10.).fit(x[ok], target[ok, t]))
        else:
            valid = np.asarray(mask).astype(bool)
            for f in range(self.fields):
                model = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15,
                    learning_rate=.05, l2_regularization=1., random_state=self.seed, early_stopping=False)
                self.models.append(model.fit(x[valid], target[..., f][valid]))
        return self

    def __call__(self, history, future, mask):
        h, f = history.detach().numpy(), future.detach().numpy()
        x = self._design(h, f)
        if self.architecture == "ar":
            location = np.stack([model.predict(x) if model is not None else self.models[23].predict(x)
                                 for model in self.models], 1)
        else:
            location = np.stack([model.predict(x.reshape(-1, x.shape[-1])).reshape(len(h), self.horizon)
                                 for model in self.models], -1)
        embedding = torch.from_numpy(h[:, -1]).to(torch.float64)
        return Forecast(torch.from_numpy(location).to(torch.float64),
                        torch.from_numpy(np.broadcast_to(self.scale, location.shape).copy()).to(torch.float64),
                        embedding, mask)


REGISTRY = {"bp": NeuralPredictor, "lstm": NeuralPredictor, "tcn": NeuralPredictor,
            "transformer": NeuralPredictor, "ar": TraditionalPredictor, "gbr": TraditionalPredictor}


def make_predictor(name, training="pto", **kwargs):
    if name not in REGISTRY:
        raise ValueError(f"unregistered predictor: {name}")
    model = REGISTRY[name](name, **kwargs)
    if training != "pto" and not model.supports_jpo:
        raise ValueError(f"{name} has no native gradient JPO; use PTO")
    return model.double() if isinstance(model, nn.Module) else model
