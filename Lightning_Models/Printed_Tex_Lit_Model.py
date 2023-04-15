from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import pytorch_lightning as pl

from Lightning_Models.metrics import CharacterErrorRate
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")

try:
    import wandb
except ModuleNotFoundError:
    pass

class LitResNetTransformer(pl.LightningModule):
    def __init__(
        self,
        model,
        WandB = True,
        lr: float = 0.0004,
        weight_decay: float = 0.0005,
        milestones: List[int] = [2,4,5,6,7,9,10,12,15,17,20,25],
        gamma: float = 0.85,
    ):
        super().__init__()



        # TODO: implement saving parameters
        #self.save_hyperparameters()  # save parameters
        self.WandB =WandB
        if self.WandB:
            wandb.init() # initiare wieghts and biases
        self.lr = lr
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.milestones = milestones
        self.gamma = gamma



        self.model = model
        start_index = int(model.sos_index)
        end_index = int(model.eos_index)
        padding_index = int(model.pad_index)
        self.ignore_tokens = {start_index, end_index, padding_index}
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=padding_index)

        # Character error functions
        self.val_cer = CharacterErrorRate(self.ignore_tokens)
        self.test_cer = CharacterErrorRate(self.ignore_tokens)

    def forward(self, x):
        return self.model.predict(x)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        imgs, targets = imgs.to(dev), targets.to(dev)
        # targets = targets.squeeze(1)
        logits = self.model(imgs, targets[:, :-1])
        loss = self.loss_fn(logits, targets[:, 1:])
        self.log("train/loss", loss, prog_bar=True)
        wandb.log({"train/loss": loss})

        outputs = {"loss": loss}

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch

        # targets = targets.squeeze(1)

        logits = self.model(imgs, targets[:, :-1])
        loss = self.loss_fn(logits, targets[:, 1:])
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        wandb.log({"validation/loss": loss})

        preds = self.model.predict(imgs)
        val_cer = self.val_cer(preds, targets)
        self.log("val/cer", val_cer,   prog_bar=True)
        wandb.log({"validation/cer": val_cer})

    def test_step(self, batch, batch_idx):
        imgs, targets = batch

        # targets = targets.squeeze(1)
        preds = self.model.predict(imgs)
        test_cer = self.test_cer(preds, targets)
        self.log("test/cer", test_cer)
        return preds

    def test_epoch_end(self, test_outputs):
        with open("test_predictions.txt", "w") as f:
            for preds in test_outputs:
                for pred in preds:
                    decoded = self.tokenizer.decode(pred.tolist())
                    decoded.append("\n")
                    decoded_str = " ".join(decoded)
                    f.write(decoded_str)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)
        return [optimizer], [scheduler]

    def configure_optimizers_new(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=0.001, total_steps=100)
        return [optimizer], [scheduler]