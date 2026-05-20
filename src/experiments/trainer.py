from pathlib import Path
from typing import Mapping, Any

from src.experiments.pipeline import run_trained_model_experiment

class Trainer:
    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)
        self.experiment_name = config["experiment"]["type"]
        self.run_name = config.get("output", {}).get("run_name", config.get("run_name", "train_run"))
        self.output_dir = Path(config["output"]["root"]) / self.experiment_name / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.current_epoch = 0
        training_config = config.get("training", {}) if isinstance(config.get("training"), Mapping) else {}
        self.n_epochs = int(training_config.get("epochs", training_config.get("n_epochs", 1)))

    def train_loop(self):
        result = run_trained_model_experiment(
            self.config,
            model_name=str(self.config.get("model", {}).get("name", "trainer_gated_ppo")),
            run_dir=str(self.output_dir),
        )
        result["trainer_epochs_requested"] = self.n_epochs
        self.current_epoch = int(result.get("training_history", [{}])[-1].get("epoch", self.n_epochs - 1)) if result.get("training_history") else self.n_epochs
        result["trainer_current_epoch"] = self.current_epoch
        return result
