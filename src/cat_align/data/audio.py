from pathlib import Path

import kaldi_native_fbank as knf
import numpy as np
import soundfile as sf
import torch
from torchaudio.functional import resample

SAMPLE_RATE = 16_000


def extract_fbank(path: str | Path, num_mel_bins: int = 80) -> tuple[torch.Tensor, float]:
    """Load audio as 16 kHz mono and return filterbank frames and duration."""
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if not samples.size:
        raise ValueError(f"Audio file is empty: {path}")

    duration = len(samples) / sample_rate
    waveform = torch.from_numpy(samples.mean(axis=1)) * 32768.0
    if sample_rate != SAMPLE_RATE:
        waveform = resample(waveform, sample_rate, SAMPLE_RATE)

    options = knf.FbankOptions()
    options.frame_opts.samp_freq = SAMPLE_RATE
    options.frame_opts.dither = 0.0
    options.frame_opts.snip_edges = False
    options.mel_opts.num_bins = num_mel_bins
    extractor = knf.OnlineFbank(options)
    extractor.accept_waveform(SAMPLE_RATE, waveform.tolist())
    extractor.input_finished()
    if not extractor.num_frames_ready:
        raise ValueError(f"Audio is too short for Kaldi feature extraction: {path}")

    features = np.stack([extractor.get_frame(i) for i in range(extractor.num_frames_ready)])
    return torch.from_numpy(features), duration
