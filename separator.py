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

    def overlap_window(self, xn: torch.Tensor) -> torch.Tensor:
        batch_size, *_ = xn.shape
        total_length = self.window_size + (batch_size - 1) * self.step_size

        yn = torch.zeros(2, total_length)

        hann_window = torch.hann_window(self.window_size)

        for index, chunk in enumerate(xn):
            start = index * self.step_size
            end = start + self.window_size
            yn[:, start:end] = yn[:, start:end] + chunk * hann_window

        return yn

    def truncate_concat(self):
        pass

    def pre_process_data(self, xn):
        """
        Pads audio, unfolds it into chunks.
        :param xn:
        :param frameSize:
        :param stepSize:
        :return:
        """
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
            padding = (self.step_size - (len(channel_data) % self.step_size)) % self.step_size
            if padding > 0:
                padded_data = torch.cat((channel_data, torch.zeros(padding)))
            else:
                padded_data = channel_data

            # compute iterations
            total_length = len(padded_data)
            iterations = (total_length - self.window_size) // self.step_size + 1

            # extract data and assign into frame
            channel_frames = torch.zeros((iterations, self.window_size))
            for i in range(iterations):
                start = i * self.step_size
                end = start + self.window_size
                channel_frames[i, :] = padded_data[start:end]

            dataSet.append(channel_frames)

        # merge list as tensor
        dataSet = torch.stack(dataSet)

        return dataSet

    @torch.no_grad()
    def forward(self, raw_audio: torch.Tensor) -> torch.Tensor:
        """
        Applies separation on the chunk level in the TF-domain,
        restores audio via
        1. overlap-add-average
        2. overlap-add-hann-window
        3. truncate-concat

        Input shape:
            [n_channels, duration]
        Output shape:
            [n_channels, duration]
        """
        duration = raw_audio.shape[1]

        y = self.pre_process_data(raw_audio)
        y = rearrange(y, 's b t -> b s t')
        y_hat = self.model(y)

        raw_output = self.overlap_window(y_hat)
        output = raw_output[:, :duration]

        return output
