# CAT Align

从 CTC-TTS 整理出的独立 Python 强制对齐包。输入一段音频及其转录，输出逐词和逐音素时间戳。通过 Whistle Conformer 生成 CTC logits，再用 TorchAudio 计算对齐路径。

## 安装

在项目根目录运行，需要 Python 3.10 或更新版本。

```bash
python -m pip install .
```

## Python Package Usage (Quick Inference)

准备本地权重 `best-10.pt`、配套词表 `tokenizer_updated.tknz`，以及一段包含 “Hello world” 的 `hello.wav`。省略 `config` 时，使用随包附带的 CTC-TTS Conformer 默认配置。默认在 CPU 上推理，可指定 `device="cuda"`。[官方对齐权重与词表](https://huggingface.co/THU-SPMI/CTC-TTS/tree/71b77cb7e5a1f2d5edb7bd5d11c7cde51de16988/multis_peaker)

```python
from cat_align import CATAligner

aligner = CATAligner(
    checkpoint="best-10.pt",
    tokenizer="tokenizer_updated.tknz",
)
words = aligner.align("hello.wav", "Hello world.")

for word in words:
    print(f"{word.word}: {word.start:.2f}–{word.end:.2f} s")
    for phone in word.phonemes:
        print(f"  {phone.phoneme}: {phone.start:.2f}–{phone.end:.2f} s")
```

也可以下载公开录音，将示例的转录换成 `I had that curiosity beside me at this moment.`。[TorchAudio 示例 WAV](https://download.pytorch.org/torchaudio/tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav)

同一个 `aligner` 可以对齐多条音频。WAV、FLAC 等音频会转为 16 kHz 单声道。转录按词查发音，保留 `don't` 这类词内的撇号。词表外单词可以通过 `pronunciations` 指定发音。

```python
words = aligner.align("name.wav", "Hi, Zobox.", pronunciations={"Zobox": "z o ʊ b ɑ k s"})
```

如果已有音素标签，可以直接传入模型词表中的音素序列。

```python
phones = aligner.align_phonemes("hello.wav", "h ɛ l o ʊ w ə ɹ l d")
```

`start` 和 `end` 的单位为秒，音素另有 `start_frame` 和 `end_frame`。区间右端不包含在内，默认模型每帧 40 ms。沿用 CTC-TTS 的 blank 帧归属规则，开头和中间的 blank 帧归后一音素，结尾的 blank 帧归最后一个音素。

## 自定义 Conformer 配置

使用其他 Conformer 权重时，传入与权重和词表配套的 JSON。可以使用完整的 CTC-TTS 配置，也可以将 `encoder` 对象中的字段直接写在 JSON 顶层。[默认配置](cat_align/models/conformer.json) · [CTC-TTS 上游代码](https://github.com/thu-spmi/CTC-TTS)

```python
aligner = CATAligner(
    checkpoint="my-conformer.pt",
    tokenizer="my-tokenizer.tknz",
    config="my-conformer.json",
)
```

```json
{
  "encoder": {
    "type": "ConformerNet",
    "tokens_per_second": 25,
    "kwargs": {
      "num_cells": 18,
      "idim": 80,
      "hdim": 512,
      "conv": "vgg2l",
      "num_heads": 8,
      "kernel_size": 32,
      "modernized_atten": true
    }
  }
}
```

`idim` 同时决定模型输入维度和 Mel 滤波器组维度，`tokens_per_second` 用于将帧转换为秒。`num_classes` 可由配套词表推导，显式填写时须与词表大小一致。音频对齐使用静态 Fbank 特征、VGG2L 下采样和 CTC 分类头。
