from ray.train import Checkpoint
from ray import train
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import tempfile
from pathlib import Path


class DartsCheckpointCallback(ModelCheckpoint):
    def __init__(self, metrics, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metrics = metrics

    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        filepath_elems = Path(filepath).parts[-3:]
        with tempfile.TemporaryDirectory() as tmpdir:
            tempdir = Path(tmpdir)
            super()._save_checkpoint(trainer, tempdir / str(Path(*filepath_elems)))

            if trainer.sanity_checking:
                return

            report_dict = self._get_report_dict(trainer)
            if not report_dict:
                return

            checkpoint = Checkpoint.from_directory(tempdir / str(Path(*filepath_elems[:-1])))
            train.report(report_dict, checkpoint=checkpoint)

    def _get_report_dict(self, trainer: Trainer):
        if trainer.sanity_checking:
            return
        if not self._metrics:
            report_dict = {k: v.item() for k, v in trainer.callback_metrics.items()}
        else:
            report_dict = {}
            for key in self._metrics:
                if isinstance(self._metrics, dict):
                    metric = self._metrics[key]
                else:
                    metric = key
                if metric in trainer.callback_metrics:
                    report_dict[key] = trainer.callback_metrics[metric].item()
                else:
                    print(
                        f"Metric {metric} does not exist in "
                        "`trainer.callback_metrics."
                    )

        return report_dict

    def set_model(self, model):
        self._model = model
