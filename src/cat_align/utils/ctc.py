from dataclasses import dataclass
from itertools import groupby, pairwise
from warnings import catch_warnings, filterwarnings

import torch
from torchaudio.functional import forced_align


@dataclass(frozen=True)
class PhonemeSpan:
    phoneme: str
    start: float
    end: float
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class WordSpan:
    word: str
    phonemes: tuple[PhonemeSpan, ...]

    @property
    def start(self) -> float:
        return self.phonemes[0].start

    @property
    def end(self) -> float:
        return self.phonemes[-1].end


def align_frames(logits: torch.Tensor, target: list[int]) -> list[tuple[int, int]]:
    """Align a transcript and return a half-open frame interval per phoneme."""
    if not target:
        raise ValueError("The transcript must contain at least one phoneme.")
    min_frames = len(target) + sum(left == right for left, right in pairwise(target))
    if min_frames > logits.size(0):
        raise ValueError(
            f"Transcript requires at least {min_frames} CTC frames, "
            f"but the audio provides {logits.size(0)}."
        )

    labels = torch.tensor([target], dtype=torch.long, device="cpu")
    log_probs = logits.float().cpu().log_softmax(dim=-1).unsqueeze(0)
    # TorchAudio 2.8 warns about a removal that was subsequently canceled.
    with catch_warnings():
        filterwarnings(
            "ignore", r"torchaudio\.functional\._alignment\.forced_align has been deprecated"
        )
        path, _ = forced_align(log_probs, labels, blank=0)

    # Assign blanks to the following phoneme, or to the last phoneme at the end.
    tokens, ends = [], []
    frame = 0
    for token, run in groupby(path[0].tolist()):
        frame += sum(1 for _ in run)
        if token != 0:
            tokens.append(token)
            ends.append(frame)
    if tokens != target:
        raise RuntimeError("CTC path did not match the transcript phonemes.")
    ends[-1] = frame
    return list(pairwise([0, *ends]))
