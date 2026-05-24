# PyTorch Lightning training script for ATISS
import argparse
import logging
import os
import sys

import numpy as np

import torch
from torch.utils.data import DataLoader, IterableDataset

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm

from training_utils import id_generator, save_experiment_params, load_config

from scene_synthesis.datasets import get_encoded_dataset, filter_function
from scene_synthesis.networks import build_network, optimizer_factory
from scene_synthesis.stats_logger import StatsLogger, WandB


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
        f for f in os.listdir(experiment_directory)
        if f.startswith("model_")
    ]
    if len(model_files) == 0:
        return
    ids = [int(f[6:]) for f in model_files]
    max_id = max(ids)
    model_path = os.path.join(
        experiment_directory, "model_{:05d}"
    ).format(max_id)
    opt_path = os.path.join(
        experiment_directory, "opt_{:05d}"
    ).format(max_id)
    if not (os.path.exists(model_path) and os.path.exists(opt_path)):
        return

    print("Loading model checkpoint from {}".format(model_path))
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Loading optimizer checkpoint from {}".format(opt_path))
    optimizer.load_state_dict(
        torch.load(opt_path, map_location=device)
    )
    args.continue_from_epoch = max_id+1


def save_checkpoints(epoch, model, optimizer, experiment_directory):
    torch.save(
        model.state_dict(),
        os.path.join(experiment_directory, "model_{:05d}").format(epoch)
    )
    torch.save(
        optimizer.state_dict(),
        os.path.join(experiment_directory, "opt_{:05d}").format(epoch)
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
                split=self.config["training"].get("splits", ["train", "val"])
            ),
            path_to_bounds=None,
            augmentations=self.config["data"].get("augmentations", None),
            split=self.config["training"].get("splits", ["train", "val"])
        )
        np.savez(
            self.path_to_bounds,
            sizes=self.train_dataset.bounds["sizes"],
            translations=self.train_dataset.bounds["translations"],
            angles=self.train_dataset.bounds["angles"]
        )
        print("Saved the dataset bounds in {}".format(self.path_to_bounds))

        self.validation_dataset = get_encoded_dataset(
            self.config["data"],
            filter_function(
                self.config["data"],
                split=self.config["validation"].get("splits", ["test"])
            ),
            path_to_bounds=self.path_to_bounds,
            augmentations=None,
            split=self.config["validation"].get("splits", ["test"])
        )

        print("Loaded {} training scenes with {} object types".format(
            len(self.train_dataset), self.train_dataset.n_object_types)
        )
        print("Training set has {} bounds".format(self.train_dataset.bounds))
        print("Loaded {} validation scenes with {} object types".format(
            len(self.validation_dataset), self.validation_dataset.n_object_types)
        )
        print("Validation set has {} bounds".format(
            self.validation_dataset.bounds)
        )

        assert self.train_dataset.object_types == \
            self.validation_dataset.object_types

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config["training"].get("batch_size", 128),
            num_workers=self.n_processes,
            collate_fn=self.train_dataset.collate_fn,
            shuffle=True
        )
        return DataLoader(ForeverDataset(dataloader), batch_size=None)

    def val_dataloader(self):
        dataloader = DataLoader(
            self.validation_dataset,
            batch_size=self.config["validation"].get("batch_size", 1),
            num_workers=self.n_processes,
            collate_fn=self.validation_dataset.collate_fn,
            shuffle=False
        )
        return DataLoader(
            ConditionalDataset(
                dataloader,
                lambda: self.run_validation
            ),
            batch_size=None
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
                self.config["training"],
                self.network.parameters()
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
            batch_size=self.config["training"].get("batch_size", 128)
        )
        return loss

    @property
    def legacy_epoch(self):
        return self.start_epoch + self.current_epoch

    def _update_epoch_metrics(self, metrics_attr, loss):
        metrics = getattr(self, metrics_attr)
        if metrics is None:
            metrics = {
                "loss": 0.0,
                "count": 0,
                "values": {},
                "last_values": {}
            }
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
            batch_size=self.config["validation"].get("batch_size", 1)
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
            prefix,
            epoch,
            metrics["loss"] / count
        )
        for key, value in sorted(metrics["values"].items()):
            msg += " - {}: {:.5f}".format(key, value / count)
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
                "train",
                epoch+1,
                pl_module.pop_epoch_metrics("_train_metrics")
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
                pl_module.legacy_epoch+1,
                pl_module.pop_epoch_metrics("_val_metrics")
            )
        )
        StatsLogger.instance().clear()
        print("====> Validation Epoch ====>")


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
            "Batches epoch {}/{}".format(current_epoch, total_epochs)
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


def main(argv):
    parser = argparse.ArgumentParser(
        description="Train a generative model on bounding boxes with Lightning"
    )

    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "output_directory",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help=("The path to a previously trained model to continue"
              " the training from")
    )
    parser.add_argument(
        "--continue_from_epoch",
        default=0,
        type=int,
        help="Continue training from epoch (default=0)"
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=0,
        help="The number of processed spawned by the batch provider"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=27,
        help="Seed for the PRNG"
    )
    parser.add_argument(
        "--experiment_tag",
        default=None,
        help="Tag that refers to the current experiment"
    )
    parser.add_argument(
        "--with_wandb_logger",
        action="store_true",
        help="Use wandB for logging the training progress"
    )

    args = parser.parse_args(argv)

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

    experiment_directory = os.path.join(
        args.output_directory,
        experiment_tag
    )
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)

    save_experiment_params(args, experiment_tag, experiment_directory)
    print("Save experiment statistics in {}".format(experiment_directory))

    config = load_config(args.config_file)

    data_module = ATISSDataModule(
        config,
        experiment_directory,
        args.n_processes
    )
    data_module.setup("fit")

    network, _, _ = build_network(
        data_module.train_dataset.feature_size,
        data_module.train_dataset.n_classes,
        config,
        args.weight_file,
        device=device
    )

    optimizer = optimizer_factory(config["training"], network.parameters())
    load_checkpoints(network, optimizer, experiment_directory, args, device)
    lightning_module = ATISSLightningModule(
        network,
        config,
        start_epoch=args.continue_from_epoch,
        optimizer=optimizer
    )

    if args.with_wandb_logger:
        WandB.instance().init(
            config,
            model=network,
            project=config["logger"].get(
                "project", "autoregressive_transformer"
            ),
            name=experiment_tag,
            watch=False,
            log_frequency=10
        )

    StatsLogger.instance().add_output_file(open(
        os.path.join(experiment_directory, "stats.txt"),
        "w"
    ))

    epochs = config["training"].get("epochs", 150)
    steps_per_epoch = config["training"].get("steps_per_epoch", 500)
    save_every = config["training"].get("save_frequency", 10)
    val_every = config["validation"].get("frequency", 100)

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=epochs-args.continue_from_epoch,
        min_epochs=epochs-args.continue_from_epoch,
        limit_train_batches=steps_per_epoch,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
        callbacks=[
            LegacyATISSCallback(experiment_directory, save_every, val_every),
            EpochTQDMProgressBar(start_epoch=args.continue_from_epoch)
        ]
    )
    trainer.fit(
        lightning_module,
        datamodule=data_module,
        ckpt_path=None
    )


if __name__ == "__main__":
    main(sys.argv[1:])
