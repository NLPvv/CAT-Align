# CAT-Align

从 CTC-TTS 整理出的独立 Python 强制对齐包。输入一段音频及其转录，输出逐词和逐音素时间戳。通过 CAT-Conformer 生成 CTC logits，再用 TorchAudio 计算对齐路径。

## 安装

在项目根目录运行，需要 Python 3.10 或更新版本。

```bash
python -m pip install .
```

## Python Package Usage (Quick Inference)

准备本地权重 `best-10.pt`、配套词表 `tokenizer_updated.tknz`。默认在 CPU 上推理，可指定 `device="cuda"`。

```python
from cat_align import CATAligner

aligner = CATAligner(
    checkpoint="best-10.pt",
    tokenizer="tokenizer_updated.tknz",
    config="conformer.json",
    device="cuda:0"
)
words = aligner.align("hello.wav", "Hello world.")

for word in words:
    print(f"{word.word}: {word.start:.2f}–{word.end:.2f} s")
    for phone in word.phonemes:
        print(f"  {phone.phoneme}: {phone.start:.2f}–{phone.end:.2f} s")
```

如果已有音素标签，可以直接传入模型词表中的音素序列。

```python
phones = aligner.align_phonemes("hello.wav", "h ɛ l o ʊ w ə ɹ l d")
```

`start` 和 `end` 的单位为秒，音素另有 `start_frame` 和 `end_frame`。区间右端不包含在内，默认模型每帧 40 ms。沿用 CTC-TTS 的 blank 帧归属规则，开头和中间的 blank 帧归后一音素，结尾的 blank 帧归最后一个音素。

