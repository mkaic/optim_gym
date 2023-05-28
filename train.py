from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.datasets import CelebA
from torchvision import transforms as T
from lightning_vae import VAEExperiment
import pytorch_lightning as pl
import torch
from pathlib import Path

print("loading datasets...")
input_size = 128
transforms = T.Compose(
    [T.RandomHorizontalFlip(), T.CenterCrop(148), T.Resize(input_size), T.ToTensor()]
)

download = not Path("./celeb_a").exists()

train_ds = CelebA(
    root="./celeb_a", split="train", download=download, transform=transforms
)
val_ds = CelebA(
    root="./celeb_a", split="valid", download=download, transform=transforms
)
test_ds = CelebA(
    root="./celeb_a", split="test", download=download, transform=transforms
)
print("datasets loaded!")

print("train_ds length: ", len(train_ds))
print("val_ds length: ", len(val_ds))
print("test_ds length: ", len(test_ds))

batch_size = 512
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

batch = next(iter(train_dl))
print(batch[0].shape)

optimizer = torch.optim.Adam

model = VAEExperiment(
    lr=1e-3,
    optim=Adam,
    input_size=input_size,
)

trainer = pl.Trainer(gpus=1, precision="bf16", max_epochs=100)
trainer.fit(model, train_dl, val_dl)
torch.save(model.state_dict(), Path(trainer.log_dir) / "final_weights.pt")
