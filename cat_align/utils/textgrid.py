from collections.abc import Iterable, Sequence
from math import isfinite

from .ctc import WordSpan


def fill_gaps(
    spans: Iterable[tuple[float, float, str]], duration: float
) -> list[tuple[float, float, str]]:
    intervals = []
    cursor = 0.0
    for start, end, label in spans:
        if start == end:
            continue
        if not cursor <= start < end <= duration:
            raise ValueError("TextGrid intervals must be ordered and inside the audio duration.")
        if cursor < start:
            intervals.append((cursor, start, ""))
        intervals.append((start, end, label))
        cursor = end
    if cursor < duration:
        intervals.append((cursor, duration, ""))
    return intervals


def format_textgrid(words: Sequence[WordSpan], duration: float) -> str:
    """Write word and phoneme tiers, omitting intervals of zero duration."""
    if not isfinite(duration) or duration <= 0:
        raise ValueError("TextGrid requires a finite, positive audio duration.")

    tiers = (
        ("words", ((word.start, word.end, word.word) for word in words)),
        ("phonemes", (
            (phone.start, phone.end, phone.phoneme)
            for word in words for phone in word.phonemes
        )),
    )
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        f"xmax = {duration}",
        "tiers? <exists>",
        f"size = {len(tiers)}",
        "item []:",
    ]
    for tier_index, (name, spans) in enumerate(tiers, 1):
        intervals = fill_gaps(spans, duration)
        lines.extend((
            f"    item [{tier_index}]:",
            '        class = "IntervalTier"',
            f'        name = "{name}"',
            "        xmin = 0",
            f"        xmax = {duration}",
            f"        intervals: size = {len(intervals)}",
        ))
        for index, (start, end, label) in enumerate(intervals, 1):
            label = label.replace('"', '""')
            lines.extend((
                f"        intervals [{index}]:",
                f"            xmin = {start}",
                f"            xmax = {end}",
                f'            text = "{label}"',
            ))
    return "\n".join(lines) + "\n"
