import typing as tp

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import Optimizer, lr_scheduler
from omegaconf import DictConfig

from beartype.typing import Tuple, Optional, List, Callable

from einops import rearrange, pack, unpack

class PLModel(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            featurizer: nn.Module,
            inverse_featurizer: nn.Module,
            augmentations: nn.Module,
            opt: Optimizer,
            sch: lr_scheduler._LRScheduler,
            hparams: DictConfig = None
    ):
        super().__init__()

        # augmentations
        self.augmentations = augmentations

        # featurizers
        self.featurizer = featurizer
        self.inverse_featurizer = inverse_featurizer

        # model
        self.model = model

        # losses
        self.mae_specR = nn.L1Loss()
        self.mae_specI = nn.L1Loss()
        self.mae_time = nn.L1Loss()

        # opts
        self.opt = opt
        self.sch = sch

        # logging
        self.save_hyperparameters(hparams)

    def training_step(
            self, batch, batch_idx
    ) -> torch.Tensor:
        """
        Input shape: [batch_size, n_sources, n_channels, time]
        """
        loss, loss_dict, usdr = self.step(batch)

        # logging
        for k in loss_dict:
            self.log(f"train/{k}", loss_dict[k].detach(), on_epoch=True, on_step=False)
        self.log("train/loss", loss.detach(), on_epoch=True, on_step=False)
        self.log("train/usdr", usdr.detach(), on_epoch=True, on_step=False)

        return loss

    def validation_step(
            self, batch, batch_idx
    ) -> torch.Tensor:
        loss, loss_dict, usdr = self.step(batch)
        # logging
        for k in loss_dict:
            self.log(f"val/{k}", loss_dict[k])
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/usdr", usdr, prog_bar=True)

        return loss

    def step(
            self, batchT: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, tp.Dict[str, torch.Tensor], torch.Tensor]:
        """
        Input shape: [batch_size, n_sources, n_channels, time]
        """
        # # augmentations
        # batchT = self.augmentations(batchT)
        #
        # # STFT
        # batchS = self.featurizer(batchT)
        # mixS, tgtS = batchS[:, 0], batchS[:, 1]
        #
        # # apply model
        # predS = self.model(mixS)
        #
        # # iSTFT
        # batchT = self.inverse_featurizer(
        #     torch.stack((predS, tgtS), dim=1)
        # )
        # predT, tgtT = batchT[:, 0], batchT[:, 1]
        #
        # # compute loss
        # loss, loss_dict = self.compute_losses(
        #     predS, tgtS,
        #     predT, tgtT
        # )
        #
        # # compute metrics
        # usdr = self.compute_usdr(predT, tgtT)

        # apply model
        mixT, tgtT = batchT[:, 0], batchT[:, 1]
        predT = self.model(mixT)

        # compute loss
        loss, loss_dict = self.compute_multi_scale_loss((4096, 2048, 1024, 512, 256), predT, tgtT)

        # compute metrics
        usdr = self.compute_usdr(predT, tgtT)

        return loss, loss_dict, usdr

    def compute_losses(
            self,
            predS: torch.Tensor,
            tgtS: torch.Tensor,
            predT: torch.Tensor,
            tgtT: torch.Tensor,
    ) -> tp.Tuple[torch.Tensor, tp.Dict[str, torch.Tensor]]:
        # frequency domain
        lossR = self.mae_specR(
            predS.real, tgtS.real
        )
        lossI = self.mae_specI(
            predS.imag, tgtS.imag
        )
        # time domain
        lossT = self.mae_time(
            predT, tgtT
        )
        loss_dict = {
            "lossSpecR": lossR,
            "lossSpecI": lossI,
            "lossTime": lossT
        }
        loss = lossR + lossI + lossT
        return loss, loss_dict

    def compute_multi_scale_loss(self, multi_stft_resolutions_window_sizes: Tuple[int, ...], pred: torch.Tensor, target: torch.Tensor):
        target = target[..., :pred.shape[-1]]  # protect against lost length on istft

        loss = F.l1_loss(pred, target)

        multi_stft_resolution_loss = 0.

        for window_size in multi_stft_resolutions_window_sizes:

            res_stft_kwargs = dict(
                n_fft = max(window_size, self.multi_stft_n_fft),  # not sure what n_fft is across multi resolution stft
                win_length = window_size,
                return_complex = True,
                window = torch.hann_window(window_size),
                **self.multi_stft_kwargs,
            )

            recon_Y = torch.stft(rearrange(pred, '... s t -> (... s) t'), **res_stft_kwargs)
            target_Y = torch.stft(rearrange(target, '... s t -> (... s) t'), **res_stft_kwargs)

            multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(recon_Y, target_Y)

        weighted_multi_resolution_loss = multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight

        total_loss = loss + weighted_multi_resolution_loss

        loss_dict = {
            "multi-stft loss": multi_stft_resolution_loss,
            "weighted multi-stft loss": weighted_multi_resolution_loss,
            "time loss": loss
        }

        return total_loss, loss_dict

    @staticmethod
    def compute_usdr(
            predT: torch.Tensor,
            tgtT: torch.Tensor,
            delta: float = 1e-7
    ) -> torch.Tensor:
        num = torch.sum(torch.square(tgtT), dim=(1, 2))
        den = torch.sum(torch.square(tgtT - predT), dim=(1, 2))
        num += delta
        den += delta
        usdr = 10 * torch.log10(num / den)
        return usdr.mean()

    def on_before_optimizer_step(
            self, *args, **kwargs
    ):
        norms = pl.utilities.grad_norm(self, norm_type=2)
        norms = dict(filter(lambda elem: '_total' in elem[0], norms.items()))
        self.log_dict(norms)

    def configure_optimizers(self):
        return [self.opt], [self.sch]

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict
