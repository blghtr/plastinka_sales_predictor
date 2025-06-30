import torch
from darts.logging import raise_if_not
from darts.utils.likelihood_models import QuantileRegression
from scipy.stats import norm


class WQuantileRegression(QuantileRegression):
    def __init__(
        self,
        q_weights=None,
        sigma_left_factor=None,
        sigma_right_factor=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if q_weights is None:
            if all([sigma_left_factor is not None, sigma_right_factor is not None]):
                q_weights = asymmetric_gaussian_weights(
                    self.quantiles,
                    peak_location=0.5,
                    sigma_left_factor=sigma_left_factor,
                    sigma_right_factor=sigma_right_factor,
                    peak_base=1.0,
                )
            else:
                q_weights = torch.linspace(0.3, 3, len(self.quantiles))

        if not isinstance(q_weights, torch.Tensor):
            q_weights = torch.tensor(q_weights)

        self.q_weights = q_weights

    def compute_loss(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        sample_weight: torch.Tensor,
    ):
        dim_q = 3
        device = model_output.device
        self.q_weights = self.q_weights.to(device)

        # test if torch model forward produces correct output and store quantiles tensor
        if self.first:
            raise_if_not(
                len(model_output.shape) == 4
                and len(target.shape) == 3
                and model_output.shape[:2] == target.shape[:2],
                "mismatch between predicted and target shape",
            )
            raise_if_not(
                model_output.shape[dim_q] == len(self.quantiles),
                "mismatch between number of predicted quantiles and target quantiles",
            )
            self.quantiles_tensor = torch.tensor(self.quantiles).to(device)
            self.first = False

        errors = target.unsqueeze(-1) - model_output
        losses = (
            torch.max(
                (self.quantiles_tensor - 1) * errors, self.quantiles_tensor * errors
            )
            * self.q_weights
        ).sum(dim=dim_q)

        if sample_weight is not None:
            losses = losses * sample_weight
        return losses.mean()


def asymmetric_gaussian_weights(
    quantiles, peak_location, sigma_left_factor, sigma_right_factor, peak_base
):
    """Генерирует асимметричные веса с факторами для сигм и базовым пиком."""
    weights = []
    for q in quantiles:
        if q <= peak_location:
            weights.append(
                peak_base
                * (
                    norm.pdf(q, peak_location, sigma_left_factor * peak_location)
                    / norm.pdf(
                        peak_location, peak_location, sigma_left_factor * peak_location
                    )
                )
            )
        else:
            weights.append(
                0.5
                + (peak_base - 0.5)
                * (
                    norm.pdf(q, peak_location, sigma_right_factor * (1 - peak_location))
                    / norm.pdf(
                        peak_location,
                        peak_location,
                        sigma_right_factor * (1 - peak_location),
                    )
                )
            )
    return weights
