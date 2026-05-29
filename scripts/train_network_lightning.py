# PyTorch Lightning training script for ATISS
import logging
import os
import sys
from dataclasses import dataclass

import fire
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm
from torch.utils.data import DataLoader, IterableDataset
from training_utils import id_generator, load_config, save_experiment_params

from scene_synthesis.datasets import filter_function, get_encoded_dataset
from scene_synthesis.networks import build_network, optimizer_factory
from scene_synthesis.stats_logger import StatsLogger, WandB


def git_commit_id():
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def flatten_dict(data, prefix=""):
    flattened = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, name))
        else:
            flattened[name] = value
    return flattened


def plot_training_curves(history, output_directory):
    if not history:
        return
    os.makedirs(output_directory, exist_ok=True)
    for key in sorted(history):
        values = history[key]
        if len(values) == 0:
            continue
        epochs, metric_values = zip(*values, strict=False)
        plt.figure()
        plt.plot(epochs, metric_values)
        plt.xlabel("epoch")
        plt.ylabel(key)
        plt.title(key)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, f"{key}.png"))
        plt.close()


class ForeverDataset(IterableDataset):
    def __init__(self, dataloader):
        super().__init__()
        self.dataloader = dataloader

    def __iter__(self):
        while True:
            for sample in self.dataloader:
                yield sample


class ConditionalDataset(IterableDataset):
    def __init__(self, dataloader, should_yield):
        super().__init__()
        self.dataloader = dataloader
        self.should_yield = should_yield

    def __iter__(self):
        if not self.should_yield():
            return
        for sample in self.dataloader:
            yield sample


def load_checkpoints(model, optimizer, experiment_directory, args, device):
    model_files = [
        f for f in os.listdir(experiment_directory) if f.startswith("model_")
    ]
    if len(model_files) == 0:
        return
    ids = [int(f[6:]) for f in model_files]
    max_id = max(ids)
    model_path = os.path.join(experiment_directory, "model_{:05d}").format(max_id)
    opt_path = os.path.join(experiment_directory, "opt_{:05d}").format(max_id)
    if not (os.path.exists(model_path) and os.path.exists(opt_path)):
        return

    print(f"Loading model checkpoint from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loading optimizer checkpoint from {opt_path}")
    optimizer.load_state_dict(torch.load(opt_path, map_location=device))
    args.continue_from_epoch = max_id + 1


def save_checkpoints(epoch, model, optimizer, experiment_directory):
    torch.save(
        model.state_dict(),
        os.path.join(experiment_directory, "model_{:05d}").format(epoch),
    )
    torch.save(
        optimizer.state_dict(),
        os.path.join(experiment_directory, "opt_{:05d}").format(epoch),
    )


class ATISSDataModule(pl.LightningDataModule):
    def __init__(self, config, experiment_directory, n_processes):
        super().__init__()
        self.config = config
        self.experiment_directory = experiment_directory
        self.n_processes = n_processes
        self.train_dataset = None
        self.validation_dataset = None
        self.path_to_bounds = os.path.join(experiment_directory, "bounds.npz")
        self.run_validation = False

    def setup(self, stage=None):
        if self.train_dataset is not None and self.validation_dataset is not None:
            return

        self.train_dataset = get_encoded_dataset(
            self.config["data"],
            filter_function(
                self.config["data"],
                split=self.config["training"].get("splits", ["train", "val"]),
            ),
            path_to_bounds=None,
            augmentations=self.config["data"].get("augmentations", None),
            split=self.config["training"].get("splits", ["train", "val"]),
        )
        np.savez(
            self.path_to_bounds,
            sizes=self.train_dataset.bounds["sizes"],
            translations=self.train_dataset.bounds["translations"],
            angles=self.train_dataset.bounds["angles"],
        )
        print(f"Saved the dataset bounds in {self.path_to_bounds}")

        self.validation_dataset = get_encoded_dataset(
            self.config["data"],
            filter_function(
                self.config["data"],
                split=self.config["validation"].get("splits", ["test"]),
            ),
            path_to_bounds=self.path_to_bounds,
            augmentations=None,
            split=self.config["validation"].get("splits", ["test"]),
        )

        print(
            f"Loaded {len(self.train_dataset)} training scenes with {self.train_dataset.n_object_types} object types"
        )
        print(f"Training set has {self.train_dataset.bounds} bounds")
        print(
            f"Loaded {len(self.validation_dataset)} validation scenes with {self.validation_dataset.n_object_types} object types"
        )
        print(f"Validation set has {self.validation_dataset.bounds} bounds")

        assert self.train_dataset.object_types == self.validation_dataset.object_types

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config["training"].get("batch_size", 128),
            num_workers=self.n_processes,
            collate_fn=self.train_dataset.collate_fn,
            shuffle=True,
        )
        return DataLoader(ForeverDataset(dataloader), batch_size=None)

    def val_dataloader(self):
        dataloader = DataLoader(
            self.validation_dataset,
            batch_size=self.config["validation"].get("batch_size", 1),
            num_workers=self.n_processes,
            collate_fn=self.validation_dataset.collate_fn,
            shuffle=False,
        )
        return DataLoader(
            ConditionalDataset(dataloader, lambda: self.run_validation), batch_size=None
        )


class ATISSLightningModule(pl.LightningModule):
    def __init__(self, network, config, start_epoch=0, optimizer=None):
        super().__init__()
        self.network = network
        self.config = config
        self.start_epoch = start_epoch
        self._optimizer = optimizer
        self.automatic_optimization = False
        self._run_validation = False
        self._train_metrics = None
        self._val_metrics = None

    def forward(self, sample):
        return self.network(sample)

    def configure_optimizers(self):
        if self._optimizer is None:
            self._optimizer = optimizer_factory(
                self.config["training"], self.network.parameters()
            )
        return self._optimizer

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()
        prediction = self.network(batch)
        loss = prediction.reconstruction_loss(batch, batch["lengths"])
        self.manual_backward(loss)
        optimizer.step()

        self._update_epoch_metrics("_train_metrics", loss.item())
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=self.config["training"].get("batch_size", 128),
        )
        return loss

    @property
    def legacy_epoch(self):
        return self.start_epoch + self.current_epoch

    def _update_epoch_metrics(self, metrics_attr, loss):
        metrics = getattr(self, metrics_attr)
        if metrics is None:
            metrics = {"loss": 0.0, "count": 0, "values": {}, "last_values": {}}
            setattr(self, metrics_attr, metrics)

        metrics["loss"] += loss
        metrics["count"] += 1
        for key, value in StatsLogger.instance()._values.items():
            previous = metrics["last_values"].get(key, 0.0)
            metrics["values"].setdefault(key, 0.0)
            metrics["values"][key] += value._value - previous
            metrics["last_values"][key] = value._value

    def pop_epoch_metrics(self, metrics_attr):
        metrics = getattr(self, metrics_attr)
        setattr(self, metrics_attr, None)
        return metrics

    def validation_step(self, batch, batch_idx):
        if not self._run_validation:
            return None
        prediction = self.network(batch)
        loss = prediction.reconstruction_loss(batch, batch["lengths"])
        self._update_epoch_metrics("_val_metrics", loss.item())
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.config["validation"].get("batch_size", 1),
        )
        return loss


class LegacyATISSCallback(Callback):
    def __init__(self, experiment_directory, save_every, val_every):
        super().__init__()
        self.experiment_directory = experiment_directory
        self.save_every = save_every
        self.val_every = val_every

    @staticmethod
    def format_metrics(prefix, epoch, metrics):
        if metrics is None or metrics["count"] == 0:
            return None

        count = metrics["count"]
        msg = "{} epoch: {} - loss: {:.5f}".format(
            prefix, epoch, metrics["loss"] / count
        )
        for key, value in sorted(metrics["values"].items()):
            msg += f" - {key}: {value / count:.5f}"
        return msg

    @staticmethod
    def write_summary(message):
        if message is None:
            return

        print(message, flush=True)
        for f in StatsLogger.instance()._output_files:
            if not f.isatty():
                print(message, flush=True, file=f)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = pl_module.legacy_epoch
        if (epoch % self.save_every) == 0:
            optimizer = trainer.optimizers[0]
            save_checkpoints(
                epoch,
                pl_module.network,
                optimizer,
                self.experiment_directory,
            )
        self.write_summary(
            self.format_metrics(
                "train", epoch + 1, pl_module.pop_epoch_metrics("_train_metrics")
            )
        )
        StatsLogger.instance().clear()

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = pl_module.legacy_epoch
        pl_module._run_validation = epoch % self.val_every == 0 and epoch > 0
        trainer.datamodule.run_validation = pl_module._run_validation

    def on_validation_epoch_start(self, trainer, pl_module):
        if trainer.sanity_checking or not pl_module._run_validation:
            return
        print("====> Validation Epoch ====>")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not pl_module._run_validation:
            return
        self.write_summary(
            self.format_metrics(
                "val",
                pl_module.legacy_epoch + 1,
                pl_module.pop_epoch_metrics("_val_metrics"),
            )
        )
        StatsLogger.instance().clear()
        print("====> Validation Epoch ====>")


class MLflowATISSCallback(Callback):
    def __init__(self, config, experiment_tag, tracking_uri, experiment_name):
        super().__init__()
        self.config = config
        self.experiment_tag = experiment_tag
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.history = {}
        self.mlflow = None

    def setup(self, trainer, pl_module, stage=None):
        import mlflow

        self.mlflow = mlflow
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.experiment_tag)
        mlflow.log_param("git_commit_id", git_commit_id())
        for key, value in flatten_dict(self.config).items():
            mlflow.log_param(key, value)

    def _record(self, name, epoch, value):
        self.history.setdefault(name, []).append((epoch, value))
        if self.mlflow is not None:
            self.mlflow.log_metric(name, value, step=epoch)

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = pl_module._train_metrics
        if metrics is None or metrics["count"] == 0:
            return
        epoch = pl_module.legacy_epoch + 1
        count = metrics["count"]
        self._record("train_loss", epoch, metrics["loss"] / count)
        for key, value in sorted(metrics["values"].items()):
            self._record(f"train_{key}", epoch, value / count)

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = pl_module._val_metrics
        if metrics is None or metrics["count"] == 0:
            return
        epoch = pl_module.legacy_epoch + 1
        count = metrics["count"]
        self._record("val_loss", epoch, metrics["loss"] / count)
        for key, value in sorted(metrics["values"].items()):
            self._record(f"val_{key}", epoch, value / count)

    def on_train_end(self, trainer, pl_module):
        plots_dir = os.path.join("plots", self.experiment_tag)
        plot_training_curves(self.history, plots_dir)
        if self.mlflow is not None:
            if os.path.exists(plots_dir):
                self.mlflow.log_artifacts(plots_dir, artifact_path="plots")
            self.mlflow.end_run()


class EpochTQDMProgressBar(TQDMProgressBar):
    def __init__(self, start_epoch=0, *args, **kwargs):
        super().__init__(*args, leave=False, **kwargs)
        self.start_epoch = start_epoch
        self.epoch_progress_bar = None

    def get_metrics(self, trainer, pl_module):
        metrics = super().get_metrics(trainer, pl_module)
        return metrics

    def init_train_tqdm(self):
        return Tqdm(
            desc="Batches",
            position=1,
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
            bar_format=self.BAR_FORMAT,
        )

    def init_epoch_tqdm(self, trainer):
        return Tqdm(
            desc="Epochs",
            total=trainer.max_epochs,
            initial=trainer.current_epoch,
            position=0,
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
            bar_format=self.BAR_FORMAT,
        )

    def on_train_start(self, trainer, pl_module):
        self.epoch_progress_bar = self.init_epoch_tqdm(trainer)
        self.train_progress_bar = self.init_train_tqdm()

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        current_epoch = self.start_epoch + trainer.current_epoch + 1
        total_epochs = self.start_epoch + trainer.max_epochs
        self.train_progress_bar.set_description(
            f"Batches epoch {current_epoch}/{total_epochs}"
        )

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        if self.epoch_progress_bar is not None:
            self.epoch_progress_bar.update(1)
            self.epoch_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        if self.epoch_progress_bar is not None:
            self.epoch_progress_bar.close()


@dataclass
class TrainingArgs:
    config_file: str
    output_directory: str
    weight_file: str | None = None
    continue_from_epoch: int = 0
    n_processes: int = 0
    seed: int = 27
    experiment_tag: str | None = None
    with_wandb_logger: bool = False
    with_mlflow_logger: bool = False
    mlflow_tracking_uri: str = "http://127.0.0.1:8080"
    mlflow_experiment_name: str = "furniture-layout-generation"


def train(
    config_file: str,
    output_directory: str,
    weight_file: str | None = None,
    continue_from_epoch: int = 0,
    n_processes: int = 0,
    seed: int = 27,
    experiment_tag: str | None = None,
    with_wandb_logger: bool = False,
    with_mlflow_logger: bool = False,
    mlflow_tracking_uri: str = "http://127.0.0.1:8080",
    mlflow_experiment_name: str = "furniture-layout-generation",
):
    args = TrainingArgs(
        config_file=config_file,
        output_directory=output_directory,
        weight_file=weight_file,
        continue_from_epoch=continue_from_epoch,
        n_processes=n_processes,
        seed=seed,
        experiment_tag=experiment_tag,
        with_wandb_logger=with_wandb_logger,
        with_mlflow_logger=with_mlflow_logger,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )

    logging.getLogger("trimesh").setLevel(logging.ERROR)

    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(np.iinfo(np.int32).max))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(np.random.randint(np.iinfo(np.int32).max))

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running code on", device)

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    if args.experiment_tag is None:
        experiment_tag = id_generator(9)
    else:
        experiment_tag = args.experiment_tag

    experiment_directory = os.path.join(args.output_directory, experiment_tag)
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)

    save_experiment_params(args, experiment_tag, experiment_directory)
    print(f"Save experiment statistics in {experiment_directory}")

    config = load_config(args.config_file)

    data_module = ATISSDataModule(config, experiment_directory, args.n_processes)
    data_module.setup("fit")

    network, _, _ = build_network(
        data_module.train_dataset.feature_size,
        data_module.train_dataset.n_classes,
        config,
        args.weight_file,
        device=device,
    )

    optimizer = optimizer_factory(config["training"], network.parameters())
    load_checkpoints(network, optimizer, experiment_directory, args, device)
    lightning_module = ATISSLightningModule(
        network, config, start_epoch=args.continue_from_epoch, optimizer=optimizer
    )

    if args.with_wandb_logger:
        WandB.instance().init(
            config,
            model=network,
            project=config["logger"].get("project", "autoregressive_transformer"),
            name=experiment_tag,
            watch=False,
            log_frequency=10,
        )

    StatsLogger.instance().add_output_file(
        open(os.path.join(experiment_directory, "stats.txt"), "w")
    )

    epochs = config["training"].get("epochs", 150)
    steps_per_epoch = config["training"].get("steps_per_epoch", 500)
    save_every = config["training"].get("save_frequency", 10)
    val_every = config["validation"].get("frequency", 100)

    callbacks = [
        LegacyATISSCallback(experiment_directory, save_every, val_every),
        EpochTQDMProgressBar(start_epoch=args.continue_from_epoch),
    ]
    if args.with_mlflow_logger:
        callbacks.append(
            MLflowATISSCallback(
                config,
                experiment_tag,
                args.mlflow_tracking_uri,
                args.mlflow_experiment_name,
            )
        )

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=epochs - args.continue_from_epoch,
        min_epochs=epochs - args.continue_from_epoch,
        limit_train_batches=steps_per_epoch,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
        callbacks=callbacks,
    )
    trainer.fit(lightning_module, datamodule=data_module, ckpt_path=None)


if __name__ == "__main__":
    fire.Fire(train)
