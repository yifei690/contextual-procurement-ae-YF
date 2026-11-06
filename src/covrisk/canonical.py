"""One sparse coefficient representation for exact and differentiable solvers."""
from dataclasses import dataclass, field
import numpy as np
import torch
from scipy import sparse

DTYPE = torch.float64


def scalar(x):
    return x.to(dtype=DTYPE, device="cpu") if isinstance(x, torch.Tensor) else torch.tensor(float(x), dtype=DTYPE)


@dataclass
class Program:
    c: torch.Tensor
    av: torch.Tensor
    b: torch.Tensor
    ev: torch.Tensor
    e: torch.Tensor
    ar: np.ndarray
    ac: np.ndarray
    er: np.ndarray
    ec: np.ndarray
    integers: np.ndarray
    names: list
    meta: dict = field(default_factory=dict)

    def matrices(self):
        n = len(self.c)
        a = sparse.coo_matrix((self.av.detach().numpy(), (self.ar, self.ac)), shape=(len(self.b), n)).tocsr()
        e = sparse.coo_matrix((self.ev.detach().numpy(), (self.er, self.ec)), shape=(len(self.e), n)).tocsr()
        return a, e

    def residuals(self, x):
        a, e = self.matrices()
        return {"inequality": float(np.maximum(a@x-self.b.detach().numpy(), 0).max(initial=0)),
                "equality": float(np.abs(e@x-self.e.detach().numpy()).max(initial=0)),
                "integrality": float(np.abs(x[self.integers]-np.round(x[self.integers])).max(initial=0))}


class Builder:
    def __init__(self):
        self.names, self.objective, self.integers = [], [], []
        self.a, self.b, self.e, self.rhs = [], [], [], []
        self.variable_scales = []

    def var(self, name, lower=None, upper=None, integer=False, objective=0):
        idx = len(self.names)
        self.names.append(name)
        self.objective.append(scalar(objective))
        self.variable_scales.append(scalar(upper) if lower is not None and float(scalar(lower).detach()) == 0
                                    and upper is not None and float(scalar(upper).detach()) > 0 else scalar(1))
        if integer:
            self.integers.append(idx)
        if lower is not None and upper is not None and not isinstance(lower, torch.Tensor) and not isinstance(upper, torch.Tensor) and lower == upper:
            self.eq({idx: 1}, lower)
        else:
            if lower is not None:
                self.le({idx: -1}, -scalar(lower))
            if upper is not None:
                self.le({idx: 1}, upper)
        return idx

    def le(self, coefficients, rhs):
        self.a.append(coefficients)
        self.b.append(scalar(rhs))

    def eq(self, coefficients, rhs):
        self.e.append(coefficients)
        self.rhs.append(scalar(rhs))

    def finish(self, meta=None):
        def collect(rows):
            rr, cc, vv = [], [], []
            for r, values in enumerate(rows):
                for c, v in values.items():
                    rr.append(r)
                    cc.append(c)
                    vv.append(scalar(v))
            return np.asarray(rr, int), np.asarray(cc, int), torch.stack(vv) if vv else torch.empty(0, dtype=DTYPE)
        ar, ac, av = collect(self.a)
        er, ec, ev = collect(self.e)
        metadata = dict(meta or {})
        metadata["variable_scales"] = torch.stack(self.variable_scales)
        return Program(torch.stack(self.objective), av, torch.stack(self.b) if self.b else torch.empty(0, dtype=DTYPE),
                       ev, torch.stack(self.rhs) if self.rhs else torch.empty(0, dtype=DTYPE),
                       ar, ac, er, ec, np.asarray(self.integers, int), self.names, metadata)


def add_to(coefficients, index, value):
    coefficients[index] = coefficients.get(index, 0)+value
