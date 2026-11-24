from pathlib import Path
from typing import Mapping, Sequence

import torch

from .data.audio import extract_fbank
from .data.tokenizer import Tokenizer
from .models import load_pretrained_encoder
from .utils.ctc import PhonemeSpan, WordSpan, align_frames


class CATAligner:
    """Align transcripts to audio using a reusable acoustic model."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        tokenizer: str | Path,
        device: str | torch.device = "cpu",
        config: str | Path | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.tokenizer = Tokenizer.load(tokenizer)
        vocab_size = len(self.tokenizer.tokens)
        if set(self.tokenizer.tokens) != set(range(vocab_size)):
            raise ValueError("The tokenizer IDs must match the encoder vocabulary.")

        self.model = load_pretrained_encoder(checkpoint, self.device, vocab_size, config)
        if self.model.delta_feats:
            raise ValueError("Audio alignment requires static filterbank features (delta_feats=false).")

    def align(
        self,
        audio: str | Path,
        transcript: str,
        *,
        pronunciations: Mapping[str, str] | None = None,
    ) -> list[WordSpan]:
        """Align one audio file to a transcript, with optional word-to-phoneme overrides."""
        words = self.tokenizer.encode_words(transcript, pronunciations=pronunciations)
        target = [token for _, tokens in words for token in tokens]
        spans = iter(self.align_ids(audio, target))
        return [WordSpan(word, tuple(next(spans) for _ in tokens)) for word, tokens in words]

    def align_phonemes(
        self, audio: str | Path, phonemes: str | Sequence[str]
    ) -> list[PhonemeSpan]:
        """Align model phoneme symbols given as a sequence or space-separated string."""
        return self.align_ids(audio, self.tokenizer.encode_phones(phonemes))

    @torch.inference_mode()
    def align_ids(self, audio: str | Path, target: list[int]) -> list[PhonemeSpan]:
        """Run the encoder and align the target phoneme IDs."""
        features, duration = extract_fbank(audio, num_mel_bins=self.model.idim)
        lengths = torch.tensor([features.size(0)], device=self.device)
        logits, output_lengths = self.model(features.unsqueeze(0).to(self.device), lengths)
        frames = align_frames(logits[0, : int(output_lengths[0])], target)
        frame_rate = self.model.frame_rate
        return [
            PhonemeSpan(
                phoneme=self.tokenizer.tokens[token],
                start=min(start / frame_rate, duration),
                end=min(end / frame_rate, duration),
                start_frame=start,
                end_frame=end,
            )
            for token, (start, end) in zip(target, frames)
        ]
