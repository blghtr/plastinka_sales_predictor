import torch
from torchmetrics import Metric, MetricCollection


def harmonic_mean(a: float, b: float, beta: float = 4.0) -> float:
    """
    Computes the harmonic mean between two values a and b.
    A beta > 1 gives more weight to 'a', beta < 1 gives more weight to 'b'.
    If either a or b is infinity, returns infinity.

    Args:
        a (float): First value.
        b (float): Second value.
        beta (float): Weighting factor.

    Returns:
        float: The harmonic mean.
    """
    if a == float("inf") or b == float("inf"):
        return float("inf")
    # Ensure inputs are not zero to avoid division by zero if beta makes one term dominant
    # This is implicitly handled by the structure, but good to be aware.
    return ((beta ** 2 + 1) * (a * b)) / (a * beta ** 2 + b + 1e-9) # Add epsilon for stability


class BaseMetricWithZeroHandling(Metric):
    """
    Base class for metrics that handle zero and non-zero actual values separately
    and apply a penalty if scores deviate from specified thresholds.
    """
    def __init__(self, q_interval=(0.05, 0.95), zero_thresh=0, nonzero_thresh=0, exp_base=2, direction=None):
        super().__init__()
        self.q_interval = q_interval
        self.zero_thresh = zero_thresh # Target threshold for score on zero-valued actuals
        self.nonzero_thresh = nonzero_thresh # Target threshold for score on non-zero-valued actuals
        self.exp_base = exp_base # Base for exponential penalty calculation
        self.penalty_sign = 1 if direction=='min' else -1 # 1 for minimization problems, -1 for maximization

        self.add_state("zero_sum", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("zero_count", torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("non_zero_sum", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("non_zero_count", torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred_series: torch.Tensor, actual_series: torch.Tensor):
        """
        Updates the metric state.

        Args:
            pred_series (torch.Tensor): Predicted series.
                                        Expected shape: (N, num_quantiles) or (N, S) where N is batch size,
                                        S is number of samples for quantile estimation.
                                        For MIC, actual_series is (N, 1) and pred_series is (N, S)
                                        For MIWS, actual_series is (N,) and pred_series is (N, S)
            actual_series (torch.Tensor): Actual series.
                                          Expected shape: (N,) for MIWS, (N,1) for MIC.
        """
        assert pred_series.ndim >= 1, f"pred_series must have at least 1 dimension, got {pred_series.ndim}"
        assert actual_series.ndim >= 1, f"actual_series must have at least 1 dimension, got {actual_series.ndim}"
        # More specific shape checks will be in the _compute_score of derived classes

        scaled_zero = 1e-6 # Threshold to consider an actual value as zero
        zero_mask = actual_series == scaled_zero # Mask for actuals that are effectively zero
        non_zero_mask = ~zero_mask # Mask for actuals that are non-zero

        # Update sub-metrics for zero and non-zero actuals separately
        self._update_sub_metric(actual_series[zero_mask], pred_series[zero_mask], zero=True)
        self._update_sub_metric(actual_series[non_zero_mask], pred_series[non_zero_mask], zero=False)

    def _update_sub_metric(self, actual, pred, zero):
        if actual.numel() == 0: # No samples for this category (zero/non-zero)
            return
        score = self._compute_score(actual, pred) # Compute the raw score using derived class logic
        if zero:
            self.zero_sum += score.sum()
            self.zero_count += score.numel()
        else:
            self.non_zero_sum += score.sum()
            self.non_zero_count += score.numel()

    def compute(self):
        """
        Computes the final metric value from accumulated state.
        Calculates scores for zero and non-zero actuals, applies penalties,
        and combines them using a harmonic mean.
        """
        # Calculate average score for zero actuals
        zero_score = self.zero_sum / self.zero_count if self.zero_count > 0 else float("inf")
        # Calculate average score for non-zero actuals
        non_zero_score = self.non_zero_sum / self.non_zero_count if self.non_zero_count > 0 else float("inf")

        # Calculate deviation from thresholds
        zero_delta = zero_score - self.zero_thresh
        non_zero_delta = non_zero_score - self.nonzero_thresh

        penalty = 0
        # Apply exponential penalty if scores deviate from thresholds in the undesired direction
        for delta, thresh in zip([zero_delta, non_zero_delta], [self.zero_thresh, self.nonzero_thresh]):
            if thresh != 0: # Only apply penalty if a threshold is set
                # Penalty is applied if the sign of delta matches the penalty_sign (e.g., for 'min' direction, if delta is positive)
                penalty += self.penalty_sign * float(torch.sign(delta) == self.penalty_sign) * (self.exp_base**abs(delta) - 1)
        
        # Combine scores using harmonic mean and add penalty. Ensure result is non-negative.
        return max(0, harmonic_mean(zero_score, non_zero_score) + penalty)

    def _compute_score(self, actual, pred):
        # This method must be implemented by derived classes.
        raise NotImplementedError


class MIWS(BaseMetricWithZeroHandling): # Mean Interval Width Score
    """
    Mean Interval Width Score (MIWS). This metric evaluates the width of the prediction
    intervals and penalizes intervals that do not contain the actual value.
    Lower MIWS is better.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _compute_score(self, actual_series: torch.Tensor, pred_series: torch.Tensor) -> torch.Tensor:
        """
        Computes the MIWS score for given actual and predicted series.

        Args:
            actual_series (torch.Tensor): Actual values.
                                          Expected shape: (N,), where N is the number of data points.
            pred_series (torch.Tensor): Predicted quantiles or samples.
                                        Expected shape: (N, S), where N is the number of data points
                                        and S is the number of quantiles/samples used for prediction.
                                        `torch.quantile` will be applied along `dim=1` (the S dimension).

        Returns:
            torch.Tensor: A tensor of scores for each data point. Shape: (N,).
        """
        assert actual_series.ndim == 1, f"actual_series expected to be 1D (N,), got shape {actual_series.shape}"
        assert pred_series.ndim == 2, f"pred_series expected to be 2D (N, S), got shape {pred_series.shape}"
        assert actual_series.shape[0] == pred_series.shape[0], \
            f"Batch size mismatch: actual_series has {actual_series.shape[0]}, pred_series has {pred_series.shape[0]}"

        # Calculate lower and upper quantiles from the predicted series
        # dim=1 means quantiles are computed across the S dimension (samples/quantiles for each data point)
        y_pred_lo = torch.quantile(pred_series, q=self.q_interval[0], dim=1) # Shape: (N,)
        y_pred_hi = torch.quantile(pred_series, q=self.q_interval[1], dim=1) # Shape: (N,)
        
        # Interval width: difference between upper and lower predicted quantiles
        interval_width = y_pred_hi - y_pred_lo # Shape: (N,)
        assert (interval_width >= 0).all(), "Interval width must be non-negative."

        # Penalty coefficients for when actual value is outside the interval
        # c_alpha_hi is for when actual_series > y_pred_hi
        c_alpha_hi = 1 / (1 - self.q_interval[1] + 1e-9) # Add epsilon for stability
        # c_alpha_lo is for when actual_series < y_pred_lo
        c_alpha_lo = 1 / (self.q_interval[0] + 1e-9)     # Add epsilon for stability

        # Calculate score based on three cases:
        # 1. Actual is below the lower quantile: score = interval_width + penalty
        # 2. Actual is above the upper quantile: score = interval_width + penalty
        # 3. Actual is within the interval: score = interval_width
        return torch.where(
            actual_series < y_pred_lo,
            interval_width + c_alpha_lo * (y_pred_lo - actual_series), # Penalty for being too low
            torch.where(
                actual_series > y_pred_hi,
                interval_width + c_alpha_hi * (actual_series - y_pred_hi), # Penalty for being too high
                interval_width, # No penalty if within interval
            ),
        )


class MIC(BaseMetricWithZeroHandling): # Mean Interval Coverage
    """
    Mean Interval Coverage (MIC). This metric measures the proportion of actual values
    that fall within the predicted interval. Higher MIC is better.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _compute_score(self, actual_series: torch.Tensor, pred_series: torch.Tensor) -> torch.Tensor:
        """
        Computes the MIC score (0 or 1 for each data point indicating coverage).

        Args:
            actual_series (torch.Tensor): Actual values.
                                          Expected shape: (N, 1) or (N,). If (N,), it will be unsqueezed.
            pred_series (torch.Tensor): Predicted quantiles or samples.
                                        Expected shape: (N, S), where N is the number of data points
                                        and S is the number of quantiles/samples used for prediction.
                                        `torch.quantile` will be applied along `dim=-1` (the S dimension).

        Returns:
            torch.Tensor: A tensor of scores (0 or 1) for each data point. Shape: (N,).
        """
        if actual_series.ndim == 1:
            actual_series = actual_series.unsqueeze(-1) # Ensure actual_series is (N, 1) for broadcasting

        assert actual_series.ndim == 2 and actual_series.shape[1] == 1, \
            f"actual_series expected to be 2D (N, 1) after potential unsqueeze, got shape {actual_series.shape}"
        assert pred_series.ndim == 2, f"pred_series expected to be 2D (N, S), got shape {pred_series.shape}"
        assert actual_series.shape[0] == pred_series.shape[0], \
            f"Batch size mismatch: actual_series has {actual_series.shape[0]}, pred_series has {pred_series.shape[0]}"

        # Calculate lower and upper quantiles from the predicted series
        # dim=-1 (equivalent to dim=1 for 2D tensor) means quantiles are computed across the S dimension.
        # keepdim=True ensures the output shape is (N, 1) for broadcasting with actual_series (N, 1).
        y_pred_lo = torch.quantile(pred_series, q=self.q_interval[0], dim=-1, keepdim=True) # Shape: (N, 1)
        y_pred_hi = torch.quantile(pred_series, q=self.q_interval[1], dim=-1, keepdim=True) # Shape: (N, 1)
        
        # Check if actual value is within the [y_pred_lo, y_pred_hi] interval
        # Result is a boolean tensor, converted to float (0.0 or 1.0). Shape: (N, 1)
        coverage = ((y_pred_lo <= actual_series) & (actual_series <= y_pred_hi)).float()
        return coverage.squeeze(-1) # Return as (N,)


class MIWS_MIC_Ratio(Metric):
    """
    A combined metric that computes the harmonic mean of MIWS and (1 / MIC).
    This balances the interval width (MIWS, lower is better) and coverage (MIC, higher is better).
    The (1 / MIC) term is used because harmonic_mean expects components where lower is better.
    A small epsilon is added to MIC in the denominator to prevent division by zero.
    """
    def __init__(self):
        super().__init__()
        # Initialize MIWS with specific thresholds and penalty direction (minimization)
        self.miws = MIWS(nonzero_thresh=0.8, zero_thresh=0.8, exp_base=6, direction='min')
        # Initialize MIC with specific thresholds and penalty direction (maximization)
        self.mic = MIC(nonzero_thresh=0.8, zero_thresh=0.8, exp_base=4, direction='max')

    def update(self, pred_series: torch.Tensor, actual_series: torch.Tensor):
        """
        Updates the state of the underlying MIWS and MIC metrics.

        Args:
            pred_series (torch.Tensor): Predicted series.
                                        Expected shape: (N, S) for MIWS/MIC.
            actual_series (torch.Tensor): Actual series.
                                          Expected shape: (N,) for MIWS, (N,1) or (N,) for MIC.
        """
        # Shape assertions are handled within the respective update methods of MIWS and MIC.
        self.miws.update(pred_series, actual_series)
        self.mic.update(pred_series, actual_series)

    def compute(self):
        """
        Computes the final MIWS/MIC ratio.
        MIWS is used directly (lower is better).
        MIC is inverted (1 / (MIC + epsilon)) so that lower is better for the harmonic mean.
        """
        miws_val = self.miws.compute()
        mic_val = self.mic.compute()
        # Harmonic mean combines MIWS and inverted MIC. Beta=2 gives more weight to MIWS.
        return harmonic_mean(miws_val, 1 / (mic_val + 1e-10), beta=2.0)


DEFAULT_METRICS = MetricCollection(
    [
        MIWS(),
        MIC(),
        MIWS_MIC_Ratio()
    ]
)
