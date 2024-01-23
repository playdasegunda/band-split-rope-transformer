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

        # module initialization
        self.model = self.initialize_modules()

        self.window_size = 131072
        self.step_size = 65536

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

    def overlap_average(self):
        pass

    def overlap_window(self):
        pass

    def truncate_concat(self):
        pass

    def pre_process_data(self, xn, frameSize, stepSize):
        # make sure xn is a tensor
        if not torch.is_tensor(xn):
            xn = torch.tensor(xn, dtype=torch.float32)

        # check dimension of xn
        if xn.dim() != 2 or xn.size(0) != 2:
            raise ValueError("Input tensor should have dimensions of (2, t)")

        # init dataSet
        channels, t = xn.shape
        dataSet = []

        # process by channel
        for channel in range(channels):
            channel_data = xn[channel]

            # zero padding
            padding = (stepSize - (len(channel_data) % stepSize)) % stepSize
            if padding > 0:
                padded_data = torch.cat((channel_data, torch.zeros(padding)))
            else:
                padded_data = channel_data

            # compute iterations
            total_length = len(padded_data)
            iterations = (total_length - frameSize) // stepSize + 1

            # extract data and assign into frame
            channel_frames = torch.zeros((iterations, frameSize))
            for i in range(iterations):
                start = i * stepSize
                end = start + frameSize
                channel_frames[i, :] = padded_data[start:end]

            dataSet.append(channel_frames)

        # merge list as tensor
        dataSet = torch.stack(dataSet)

        return dataSet

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
        duration = raw_audio.shape[1]

        y = self.pre_process_data(raw_audio, self.window_size, self.step_size)
        y = rearrange(y, 's b t -> b s t')
        y_hat = self.model(y)

        return raw_audio
