from typing import Optional

import torch
from darts.models import TiDEModel
from darts.models.forecasting.tide_model import _TideModule  # type: ignore


class _MultiSampleTideModule(_TideModule):
    """Extension of internal `_TideModule` that computes metrics on **multiple** samples.

    Only `_update_metrics` is overridden; everything else (forward pass, loss, etc.)
    remains identical.  This keeps the behaviour of TiDE intact while allowing
    custom metrics (e.g. MIWS / MIC) that require an (N, S) tensor instead of the
    default point forecast.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        future_cov_dim: int,
        static_cov_dim: int,
        nr_params: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        decoder_output_dim: int,
        hidden_size: int,
        temporal_decoder_hidden: int,
        temporal_width_past: int,
        temporal_width_future: int,
        use_layer_norm: bool,
        dropout: float,
        temporal_hidden_size_past: Optional[int] = None,
        temporal_hidden_size_future: Optional[int] = None,
        num_samples_for_metrics: int = 100,
        **kwargs,
    ):
        # Store the custom parameter before calling super().__init__
        self.num_samples_for_metrics = num_samples_for_metrics
        
        # Call parent constructor with all required arguments
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            future_cov_dim=future_cov_dim,
            static_cov_dim=static_cov_dim,
            nr_params=nr_params,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            decoder_output_dim=decoder_output_dim,
            hidden_size=hidden_size,
            temporal_decoder_hidden=temporal_decoder_hidden,
            temporal_width_past=temporal_width_past,
            temporal_width_future=temporal_width_future,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            temporal_hidden_size_past=temporal_hidden_size_past,
            temporal_hidden_size_future=temporal_hidden_size_future,
            **kwargs,
        )

    # ---------------------------------------------------------------------
    # Core patch: use many samples when updating torchmetrics during training
    # ---------------------------------------------------------------------
    def _update_metrics(self, output: torch.Tensor, target: torch.Tensor, metrics):
        """Override original method to feed *multiple* samples to custom metrics."""
        if not len(metrics):
            return

        # Build prediction tensor of shape (N, S)
        if self.likelihood is not None:
            # Draw multiple samples from the likelihood
            samples = [self.likelihood.sample(output) for _ in range(self.num_samples_for_metrics)]
            pred = torch.stack(samples, dim=-1)  # (*batch_dims, S)
        else:
            # Deterministic model – repeat the single output S times
            pred = output.squeeze(dim=-1).unsqueeze(-1).repeat_interleave(
                self.num_samples_for_metrics, dim=-1
            )

        # Flatten everything except sample dimension so metrics receive (N, S)
        pred_flat = pred.reshape(-1, pred.shape[-1])
        target_flat = target.reshape(-1)

        metrics.update(pred_flat, target_flat)


class CustomTiDEModel(TiDEModel):
    """TiDEModel variant that uses _MultiSampleTideModule internally and therefore
    needs to expose the same constructor signature as the base ``TiDEModel`` so that
    Darts' internal validation does not reject legitimate kwargs.
    """

    def __init__(
        self,
        # --- original TiDEModel parameters (copied) ---
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        decoder_output_dim: int = 16,
        hidden_size: int = 128,
        temporal_width_past: int = 4,
        temporal_width_future: int = 4,
        temporal_hidden_size_past: int | None = None,
        temporal_hidden_size_future: int | None = None,
        temporal_decoder_hidden: int = 32,
        use_layer_norm: bool = False,
        dropout: float = 0.1,
        use_static_covariates: bool = True,
        # --- new param ---
        num_samples_for_metrics: int = 100,
        # --- everything else collected in kwargs (loss_fn, likelihood, etc.) ---
        **kwargs,
    ):
        self._num_samples_for_metrics = num_samples_for_metrics

        # call parent constructor with original parameters
        super().__init__(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            decoder_output_dim=decoder_output_dim,
            hidden_size=hidden_size,
            temporal_width_past=temporal_width_past,
            temporal_width_future=temporal_width_future,
            temporal_hidden_size_past=temporal_hidden_size_past,
            temporal_hidden_size_future=temporal_hidden_size_future,
            temporal_decoder_hidden=temporal_decoder_hidden,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            use_static_covariates=use_static_covariates,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Override factory to return our patched PL module
    # ------------------------------------------------------------------
    def _create_model(self, train_sample):  # type: ignore[override]
        # The body is identical to TiDEModel._create_model except for the final
        # instantiation class (we call _MultiSampleTideModule instead of _TideModule).
        (
            past_target,
            past_covariates,
            historic_future_covariates,
            future_covariates,
            static_covariates,
            future_target,
        ) = train_sample

        input_dim = (
            past_target.shape[1]
            + (past_covariates.shape[1] if past_covariates is not None else 0)
            + (
                historic_future_covariates.shape[1]
                if historic_future_covariates is not None
                else 0
            )
        )
        output_dim = future_target.shape[1]
        future_cov_dim = future_covariates.shape[1] if future_covariates is not None else 0
        static_cov_dim = (
            static_covariates.shape[0] * static_covariates.shape[1] if static_covariates is not None else 0
        )
        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters

        return _MultiSampleTideModule(
            input_dim=input_dim,
            output_dim=output_dim,
            future_cov_dim=future_cov_dim,
            static_cov_dim=static_cov_dim,
            nr_params=nr_params,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            decoder_output_dim=self.decoder_output_dim,
            hidden_size=self.hidden_size,
            temporal_decoder_hidden=self.temporal_decoder_hidden,
            temporal_width_past=self.temporal_width_past,
            temporal_width_future=self.temporal_width_future,
            use_layer_norm=self.use_layer_norm,
            dropout=self.dropout,
            temporal_hidden_size_past=self.temporal_hidden_size_past,
            temporal_hidden_size_future=self.temporal_hidden_size_future,
            num_samples_for_metrics=self._num_samples_for_metrics,
            **self.pl_module_params,
        )

    # convenience alias
    MultiSampleModule = _MultiSampleTideModule

# Public re-export for easier imports
tiDEModelWithMultiSample = CustomTiDEModel

__all__ = [
    "CustomTiDEModel",
    "tiDEModelWithMultiSample",
]
