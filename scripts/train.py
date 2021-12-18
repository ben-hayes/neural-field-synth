import auraloss
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch

from neural_field_synth.data import NSynthDataset
from neural_field_synth.models import NeuralFieldSynth, LightningWrapper


def get_model(cfg: DictConfig):
    model = NeuralFieldSynth(**cfg)
    return model


def get_loss(cfg: DictConfig):
    if cfg.loss_fn == "multiscale_stft":
        loss_fn = auraloss.freq.MultiResolutionSTFTLoss
    return loss_fn(**cfg.params)


def get_dataset_and_dataloader(dataset_cfg: DictConfig, dataloader_cfg: DictConfig):
    dataset = NSynthDataset(**dataset_cfg)
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_cfg)
    return dataset, dataloader


def wrap_model(cfg: DictConfig, model, loss_fn):
    wrapped_model = LightningWrapper(model, loss_fn, **cfg)
    return wrapped_model


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    model = get_model(cfg.model)
    loss_fn = get_loss(cfg.loss)

    wrapped_model = wrap_model(cfg.optimizer, model, loss_fn)

    _, dataloader = get_dataset_and_dataloader(**cfg.data)

    logger = WandbLogger(project="wavespace")
    checkpoint_callback = ModelCheckpoint(monitor="loss")
    print(cfg.trainer)
    trainer = pl.Trainer(**cfg.trainer, logger=logger, callbacks=[checkpoint_callback])

    trainer.fit(wrapped_model, dataloader)


if __name__ == "__main__":
    main()
