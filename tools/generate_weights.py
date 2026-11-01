"""Generate the plan's headerless little-endian FP32 model, in contract order."""
import argparse
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
D, HEADS, CONTEXT, VOCAB = 384, 6, 512, 256
PARAMETERS = 11_040_768
LAYER_SHAPES = [
    ('ln1w', (D,)), ('ln1b', (D,)), ('qkvW', (3 * D, D)),
    ('qkvB', (3 * D,)), ('oW', (D, D)), ('oB', (D,)),
    ('ln2w', (D,)), ('ln2b', (D,)), ('ffW1', (4 * D, D)),
    ('ffB1', (4 * D,)), ('ffW2', (D, 4 * D)), ('ffB2', (D,)),
]


def layout():
    yield 'tokenEmb', (VOCAB, D)
    yield 'posEmb', (CONTEXT, D)
    for layer in range(6):
        for name, shape in LAYER_SHAPES:
            yield f'layers.{layer}.{name}', shape
    yield 'finalLnW', (D,)
    yield 'finalLnB', (D,)
    yield 'lmHeadW', (VOCAB, D)


def generate():
    generator = torch.Generator(device='cpu').manual_seed(42)
    for name, shape in layout():
        field = name.split('.')[-1]
        if field in ('ln1w', 'ln2w', 'finalLnW'):
            tensor = torch.ones(shape)
        elif field in ('ln1b', 'ln2b', 'finalLnB', 'qkvB', 'oB', 'ffB1', 'ffB2'):
            tensor = torch.zeros(shape)
        else:
            tensor = torch.randn(shape, generator=generator) * 0.02
        yield name, tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'weights.bin')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open('wb') as stream:
        for _, tensor in generate():
            array = tensor.numpy().astype('<f4', copy=False)
            stream.write(array.tobytes(order='C'))
            count += array.size
    assert count == PARAMETERS
    assert args.output.stat().st_size == PARAMETERS * 4
    print(f'Generated {args.output}: {count:,} parameters, {count * 4:,} bytes (seed 42)')


if __name__ == '__main__':
    main()
