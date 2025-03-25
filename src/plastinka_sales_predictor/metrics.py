import torch
from torchmetrics import Metric, MetricCollection


def harmonic_mean(a, b, beta=4):
    if a == float("inf") or b == float("inf"):
        return float("inf")
    return ((beta ** 2 + 1) * (a * b)) / (a * beta ** 2 + b)


class BaseMetricWithZeroHandling(Metric):
    def __init__(self, q_interval=(0.05, 0.95), zero_thresh=0, nonzero_thresh=0, exp_base=2, direction=None):
        super().__init__()
        self.q_interval = q_interval
        self.zero_thresh = zero_thresh
        self.nonzero_thresh = nonzero_thresh
        self.exp_base = exp_base
        self.penalty_sign = 1 if direction=='min' else -1

        self.add_state("zero_sum", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("zero_count", torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("non_zero_sum", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("non_zero_count", torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred_series: torch.Tensor, actual_series: torch.Tensor):
        scaled_zero = 1e-6
        zero_mask = actual_series == scaled_zero
        non_zero_mask = ~zero_mask

        self._update_sub_metric(actual_series[zero_mask], pred_series[zero_mask], zero=True)
        self._update_sub_metric(actual_series[non_zero_mask], pred_series[non_zero_mask], zero=False)

    def _update_sub_metric(self, actual, pred, zero):
        if actual.numel() == 0:
            return
        score = self._compute_score(actual, pred)
        if zero:
            self.zero_sum += score.sum()
            self.zero_count += score.numel()
        else:
            self.non_zero_sum += score.sum()
            self.non_zero_count += score.numel()

    def compute(self):
        zero_score = self.zero_sum / self.zero_count if self.zero_count > 0 else float("inf")
        non_zero_score = self.non_zero_sum / self.non_zero_count if self.non_zero_count > 0 else float("inf")
        zero_delta, non_zero_delta = zero_score - self.zero_thresh, non_zero_score - self.nonzero_thresh
        penalty = 0
        for delta, thresh in zip([zero_delta, non_zero_delta], [self.zero_thresh, self.nonzero_thresh]):
            if thresh != 0:
                penalty += self.penalty_sign * float(torch.sign(delta) == self.penalty_sign) * (self.exp_base**abs(delta) - 1)
        return max(0, harmonic_mean(zero_score, non_zero_score) + penalty)

    def _compute_score(self, actual, pred):
        raise NotImplementedError


class MIWS(BaseMetricWithZeroHandling):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _compute_score(self, actual_series, pred_series):
        y_pred_lo, y_pred_hi = torch.quantile(pred_series, q=self.q_interval[0]), torch.quantile(pred_series,
                                                                                                 q=self.q_interval[1])
        interval_width = y_pred_hi - y_pred_lo

        c_alpha_hi = 1 / (1 - self.q_interval[1])
        c_alpha_lo = 1 / self.q_interval[0]

        return torch.where(
            actual_series < y_pred_lo,
            interval_width + c_alpha_lo * (y_pred_lo - actual_series),
            torch.where(
                actual_series > y_pred_hi,
                interval_width + c_alpha_hi * (actual_series - y_pred_hi),
                interval_width,
            ),
        )


class MIC(BaseMetricWithZeroHandling):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _compute_score(self, actual_series, pred_series):
        y_pred_lo = torch.quantile(pred_series, q=self.q_interval[0], dim=-1, keepdim=True)
        y_pred_hi = torch.quantile(pred_series, q=self.q_interval[1], dim=-1, keepdim=True)
        return ((y_pred_lo <= actual_series) & (actual_series <= y_pred_hi)).float()


class MIWS_MIC_Ratio(Metric):
    def __init__(self):
        super().__init__()
        self.miws = MIWS(nonzero_thresh=0.8, zero_thresh=0.8, exp_base=6, direction='min')
        self.mic = MIC(nonzero_thresh=0.8, zero_thresh=0.8, exp_base=4, direction='max')

    def update(self, pred_series: torch.Tensor, actual_series: torch.Tensor):
        self.miws.update(pred_series, actual_series)
        self.mic.update(pred_series, actual_series)

    def compute(self):
        return harmonic_mean(self.miws.compute(), 1 / (self.mic.compute() + 1e-10), 2) 


DEFAULT_METRICS = MetricCollection(
    [
        MIWS(),
        MIC(),
        MIWS_MIC_Ratio()
    ]
)
