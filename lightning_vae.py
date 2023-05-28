import os

from torch import Tensor
from backbone import VanillaVAE
import pytorch_lightning as pl
import torchvision.utils as vutils
from pathlib import Path

# Modified and adapted from https://github.com/AntixK/PyTorch-VAE


class VAEExperiment(pl.LightningModule):
    def __init__(self, optim, lr, input_size, hidden_dims=None) -> None:
        super().__init__()

        self.model = VanillaVAE(input_size=input_size, hidden_dims=hidden_dims)
        self.optim = optim
        self.lr = lr
        self.curr_device = None
        self.hold_graph = False

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(
            *results,
            M_N=0.00025,  # al_img.shape[0]/ self.num_train_imgs,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx,
        )

        self.log_dict(
            {key: val.item() for key, val in train_loss.items()}, sync_dist=True
        )

        return train_loss["loss"]

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        val_loss = self.model.loss_function(
            *results,
            M_N=1.0,  # real_img.shape[0]/ self.num_val_imgs,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx,
        )

        self.log_dict(
            {f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True
        )

    def on_validation_end(self) -> None:
        self.sample_images()

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.trainer.val_dataloaders[0]))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        #         test_input, test_label = batch
        recons = self.model.generate(test_input, labels=test_label)

        reconstructions_path = Path("lightning_logs") / "Reconstructions"
        reconstructions_path.mkdir(exist_ok=True)
        vutils.save_image(
            recons.data,
            reconstructions_path
            / f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png",
            normalize=True,
            nrow=12,
        )

        try:
            samples_path = Path("lightning_logs") / "Samples"
            samples_path.mkdir(exist_ok=True)
            samples = self.model.sample(144, self.curr_device, labels=test_label)
            vutils.save_image(
                samples.cpu().data,
                samples_path / f"{self.logger.name}_Epoch_{self.current_epoch}.png",
                normalize=True,
                nrow=12,
            )
        except Warning:
            pass

    def configure_optimizers(self):
        optimizer = self.optim(
            self.model.parameters(),
            lr=self.lr,
        )
        return optimizer
