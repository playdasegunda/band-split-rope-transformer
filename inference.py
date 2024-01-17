import argparse
import typing as tp
from pathlib import Path

import torch
import soundfile as sf
from omegaconf import OmegaConf

from data import EvalSourceSeparationDataset

class InferenceProgram:
    pass