import typing as tp
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig, OmegaConf

from einops import rearrange, pack, unpack

from train import initialize_model, initialize_featurizer
from utils.utils_inference import load_pl_state_dict, get_minibatch


class Separator(nn.Module):
    def __init__(
            self,
            cfg: DictConfig,
            ckpt_path: tp.Optional[str] = None,
    ):
        super(Separator, self).__init__()
        self.cfg = cfg

        # modules params
        self.ckpt_path = Path(ckpt_path) if ckpt_path is not None else None

        print(123123)

        # module initialization
        self.model = self.initialize_modules()


    def initialize_modules(self) -> nn.Module:

        model, *_ = initialize_model(self.cfg)
        _ = model.eval()

        # load checkpoint
        if self.ckpt_path is not None:
            if self.ckpt_path.suffix == '.ckpt':
                state_dict = load_pl_state_dict(self.ckpt_path, device='cpu')
            elif self.ckpt_path.suffix == '.pt':
                state_dict = torch.load(self.ckpt_path, map_location='cpu')
            else:
                raise ValueError(f"Expected checkpoint path, got {self.ckpt_path}.")
            _ = model.load_state_dict(state_dict, strict=True)

        return model

    def chunk(self, y: torch.Tensor):
        """
        Input shape: [n_channels, n_frames]
        Output shape: []
        """
        y = y.unfold(-1, self.window_size, self.step_size)
        y = y.chunk(self.n_chunks_per_segment, dim=-1)
        y = torch.stack(y, dim=-2)
        return y

    def overlap_average(self):
        pass

    def truncate_concat(self):
        pass

    def separate(self, y: torch.Tensor) -> torch.Tensor:
        n_chunks = y.shape[0]
        window = self.window.to(y) if self.window is not None else None

        chunks = []
        for s, e in get_minibatch(n_chunks, self.bs):
            # apply the model
            chunk = self.model(y[s:e])

            if window is None:
                chunk /= (self.ws / self.hs)
            else:
                chunk = chunk * window

            chunks.append(chunk)

        return torch.cat(chunks)

    @torch.no_grad()
    def forward(self, raw_audio: torch.Tensor) -> torch.Tensor:
        """
        Pads audio, unfolds it into even chunks, applies separation on the chunk level in the TF-domain,
        restores audio via overlap-add.
        Input shape:
            [n_channels, duration]
        Output shape:
            [n_channels, duration]
        """


        return raw_audio
