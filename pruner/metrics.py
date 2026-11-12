"""Token-selection metrics: how the Pruner's choice relates to the foreground mask and to small objects."""
import torch
import torch.nn.functional as F

from .data import BINS, GROUPS, TOKEN_PX

KEEP_RATIOS = (0.75, 0.5, 0.25)
FRAC_EDGES = (0.0, 0.25, 0.5, 0.75)  # buckets of per-cell foreground pixel fraction: 0, (0,.25], ..., (.75,1]


def cell_fg_fraction(fg):
    """fg: (..., H, W) bool pixel mask -> (..., H/32, W/32) fraction of foreground pixels per token cell."""
    lead = fg.shape[:-2]
    x = F.avg_pool2d(fg.reshape(-1, 1, *fg.shape[-2:]).float(), TOKEN_PX)
    return x.reshape(*lead, *x.shape[-2:])


class SelectionMetrics:
    """Accumulates over windows. All inputs are per window batch; images with valid=False are skipped
    for mask-based metrics (their masks are unreliable) but still compete in window-level Top-K."""

    def __init__(self):
        self.n = {}

    def _add(self, name, num, den):
        a = self.n.setdefault(name, [0.0, 0.0])
        a[0] += float(num); a[1] += float(den)

    @torch.no_grad()
    def update(self, s, fg, valid, obj):
        """s: (B, I*T) scores, fg: (B, I, H, W) bool, valid: (B, I) bool, obj: (B, I, G, K, gh, gw) bool."""
        b, i = valid.shape
        frac = cell_fg_fraction(fg).reshape(b, i, -1)  # (B, I, T)
        s = s.reshape(b, i, -1).float()
        sel = s > 0
        v = valid[..., None].expand_as(sel)
        gt = frac >= 0.5
        self._add("sel_ratio", sel[v].sum(), v.sum())
        self._add("gt_fg_ratio", gt[v].sum(), v.sum())
        self._add("fg_precision", (sel & gt & v).sum(), (sel & v).sum())
        self._add("fg_recall", (sel & gt & v).sum(), (gt & v).sum())
        # selection rate by how much of the cell is foreground (sub-token objects live in the low buckets)
        bucket = torch.bucketize(frac, torch.tensor(FRAC_EDGES, device=frac.device), right=False)
        for k in range(len(FRAC_EDGES) + 1):
            m = (bucket == k) & v
            self._add(f"sel_by_cellfrac/{k}", sel[m].sum(), m.sum())
        # object-cell retention under inference-style window-level Top-K
        flat = s.reshape(b, -1)
        o = obj.reshape(b, i, len(GROUPS), len(BINS), -1) & valid[:, :, None, None, None]
        for r in KEEP_RATIOS:
            k = int(flat.shape[1] * r)
            thr = flat.topk(k, dim=1).values[:, -1]
            kept = (flat >= thr[:, None]).reshape(b, i, 1, 1, -1)
            for gi, g in enumerate(GROUPS):
                for bi, bn in enumerate(BINS):
                    self._add(f"keep{int(r * 100)}/{g}/{bn}", (kept & o)[:, :, gi, bi].sum(), o[:, :, gi, bi].sum())
            self._add(f"keep{int(r * 100)}/gt_fg", (kept.reshape(b, i, -1) & gt & v).sum(), (gt & v).sum())

    def compute(self):
        return {k: (a / d if d > 0 else float("nan")) for k, (a, d) in self.n.items()} | \
               {"count/" + k: d for k, (a, d) in self.n.items() if k.startswith("keep50/")}
