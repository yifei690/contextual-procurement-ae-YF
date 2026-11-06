"""Fixed topology with distinct pre-decision and post-realization nodes."""
from dataclasses import dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class ScenarioTree:
    pre: np.ndarray
    post: np.ndarray

    @classmethod
    def regular(cls, horizon, branching=(3, 3), reveal=None):
        """First uncertainty arrives after action 0; later reveals are explicit.

        Node numbers encode only shared observed prefixes. The projection of
        paths MUST be applied before coefficients or transport costs are built.
        """
        if reveal is None:
            reveal = np.linspace(0, horizon, len(branching), endpoint=False, dtype=int)
        if len(reveal) != len(branching) or list(reveal) != sorted(set(reveal)):
            raise ValueError("invalid reveal schedule")
        if min(reveal, default=0) < 0 or max(reveal, default=0) >= horizon:
            raise ValueError("reveal outside horizon")
        leaves = int(np.prod(branching))
        pre, post = np.zeros((leaves, horizon), int), np.zeros((leaves, horizon), int)
        ids = np.arange(leaves)
        for t in range(horizon):
            np_pre = int(np.prod([b for b, r in zip(branching, reveal) if r < t]))
            np_post = int(np.prod([b for b, r in zip(branching, reveal) if r <= t]))
            pre[:, t] = ids // (leaves//np_pre)
            post[:, t] = ids // (leaves//np_post)
        tree = cls(pre, post)
        tree.validate()
        return tree

    def validate(self):
        if self.pre.shape != self.post.shape or self.pre.ndim != 2:
            raise ValueError("invalid tree dimensions")
        for t in range(self.pre.shape[1]):
            for node in np.unique(self.post[:, t]):
                idx = self.post[:, t] == node
                if len(np.unique(self.pre[idx, t])) != 1:
                    raise ValueError("post node merges different decision histories")
            if t:
                for node in np.unique(self.pre[:, t]):
                    idx = self.pre[:, t] == node
                    if len(np.unique(self.post[idx, t-1])) != 1:
                        raise ValueError("decision node merges different past histories")

    def project(self, paths):
        """Fixed, differentiable aggregation enforces common observable prefixes."""
        if tuple(paths.shape[:2]) != self.post.shape:
            raise ValueError("paths do not match tree")
        periods = []
        for t in range(paths.shape[1]):
            means = {int(n): paths[self.post[:, t] == n, t].mean(0) for n in np.unique(self.post[:, t])}
            periods.append(torch.stack([means[int(n)] for n in self.post[:, t]]))
        return torch.stack(periods, dim=1)

    def check_paths(self, paths, atol=1e-9):
        x = paths.detach().numpy() if isinstance(paths, torch.Tensor) else np.asarray(paths)
        if x.shape[:2] != self.post.shape:
            raise ValueError("paths do not match tree")
        for t in range(x.shape[1]):
            for node in np.unique(self.post[:, t]):
                vals = x[self.post[:, t] == node, t]
                if np.max(np.abs(vals-vals[0])) > atol:
                    raise ValueError("unshared prefix: future path information would leak")
        # Distinct decision nodes must be distinguishable from *observable*
        # past L/G. Delayed settlement prices never justify a split.
        for t in range(1, x.shape[1]):
            representatives = [np.flatnonzero(self.pre[:, t] == node)[0] for node in np.unique(self.pre[:, t])]
            for i, left in enumerate(representatives):
                for right in representatives[i+1:]:
                    if np.max(np.abs(x[left, :t, :2]-x[right, :t, :2])) <= atol:
                        raise ValueError("decision nodes distinguished only by unavailable information")
