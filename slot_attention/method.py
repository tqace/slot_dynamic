import pytorch_lightning as pl
import torch
from torch import optim
from torchvision import utils as vutils
import ipdb
from slot_attention.model import SlotAttentionModel
from slot_attention.params import SlotAttentionParams
from slot_attention.utils import Tensor
from slot_attention.utils import to_rgb_from_tensor


class SlotAttentionMethod(pl.LightningModule):
    def __init__(self, model: SlotAttentionModel, datamodule: pl.LightningDataModule, params: SlotAttentionParams):
        super().__init__()
        self.model = model
        self.datamodule = datamodule
        self.params = params

    def forward(self, input, **kwargs):
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        train_loss = self.model.loss_function(batch)
        logs = {key: val.item() for key, val in train_loss.items()}
        self.log_dict(logs, sync_dist=True)
        return train_loss

    def sample_images(self):
        dl = self.datamodule.val_dataloader()
        #perm = torch.randperm(self.params.val_batch_size)
        #idx = perm[: self.params.n_samples]
        batch = next(iter(dl))
        batch = batch.to(self.device)
        recon_combined, recons, masks, slots, recon_combined_preds, recons_preds, masks_preds, slots_preds = self.model.forward(batch)
        # combine images in a nice way so we can display all outputs in one grid, output rescaled to be between 0 and 1
        B,L,C,H,W=batch.shape
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    batch[:,[4,9],:,:,:].view(B*2,C,H,W).unsqueeze(1),  # original images
                    batch[:,-2:,:,:,:].view(B*2,C,H,W).unsqueeze(1),  # original images
                    #recon_combined[:,-2:,:,:,:].view(B*2,C,H,W).unsqueeze(1),  # reconstructions
                    recon_combined_preds.view(B*2,C,H,W).unsqueeze(1),
                    (recons_preds* masks_preds + (1 - masks_preds)).view(B*2,-1,C,H,W),  # each slot
                ],
                dim=1,
            )
        )

        batch_size, max_len,num_slots, C, H, W = recons.shape
        images = vutils.make_grid(
            out.view(batch_size * 2 * out.shape[1], C, H, W).cpu(), normalize=False, nrow=out.shape[1],
        )

        return images

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        val_loss = self.model.loss_function(batch)
        return val_loss

    def validation_epoch_end(self, outputs):
        avg_loss_pred = torch.stack([x["loss_pred"] for x in outputs]).mean()
        avg_loss_recon = torch.stack([x["loss_recon"] for x in outputs]).mean()
        logs = {
            "avg_val_loss_recon": avg_loss_recon,
            "avg_val_loss_pred": avg_loss_pred,
        }
        self.log_dict(logs, sync_dist=True)
        print("; ".join([f"{k}: {v.item():.6f}" for k, v in logs.items()]))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)

        warmup_steps_pct = self.params.warmup_steps_pct
        decay_steps_pct = self.params.decay_steps_pct
        total_steps = self.params.max_epochs * len(self.datamodule.train_dataloader())

        def warm_and_decay_lr_scheduler(step: int):
            warmup_steps = warmup_steps_pct * total_steps
            decay_steps = decay_steps_pct * total_steps
            assert step < total_steps
            if step < warmup_steps:
                factor = step / warmup_steps
            else:
                factor = 1
            factor *= self.params.scheduler_gamma ** (step / decay_steps)
            return factor

        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_and_decay_lr_scheduler)

        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "step",}],
        )
