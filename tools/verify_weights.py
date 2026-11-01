"""Validate every tensor against seed 42 and optionally export a PyTorch oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from generate_weights import ROOT, D, HEADS, PARAMETERS, layout, generate


def read_weights(path):
    if path.stat().st_size != PARAMETERS * 4:
        raise ValueError(f'Expected {PARAMETERS * 4} bytes')
    flat = np.fromfile(path, dtype='<f4')
    weights, offset = {}, 0
    for name, shape in layout():
        count = int(np.prod(shape))
        value = flat[offset:offset + count].reshape(shape)
        if not np.isfinite(value).all():
            raise ValueError(f'Non-finite values in {name}')
        weights[name] = torch.from_numpy(value.copy())
        offset += count
    assert offset == PARAMETERS
    return weights


def reference(weights, tokens):
    x = weights['tokenEmb'][tokens] + weights['posEmb'][:len(tokens)]
    for layer in range(6):
        w = {name: weights[f'layers.{layer}.{name}'] for name, _ in __import__('generate_weights').LAYER_SHAPES}
        h = F.layer_norm(x, (D,), w['ln1w'], w['ln1b'], 1e-5)
        q, k, v = F.linear(h, w['qkvW'], w['qkvB']).chunk(3, dim=-1)
        q, k, v = [t.reshape(-1, HEADS, D // HEADS).transpose(0, 1) for t in (q, k, v)]
        scores = q @ k.transpose(-1, -2) / (D // HEADS) ** 0.5
        mask = torch.ones(len(tokens), len(tokens), dtype=torch.bool).triu(1)
        attention = torch.softmax(scores.masked_fill(mask, float('-inf')), dim=-1) @ v
        x = x + F.linear(attention.transpose(0, 1).reshape(-1, D), w['oW'], w['oB'])
        h = F.layer_norm(x, (D,), w['ln2w'], w['ln2b'], 1e-5)
        x = x + F.linear(F.gelu(F.linear(h, w['ffW1'], w['ffB1']), approximate='tanh'), w['ffW2'], w['ffB2'])
    h = F.layer_norm(x[-1], (D,), weights['finalLnW'], weights['finalLnB'], 1e-5)
    return F.linear(h, weights['lmHeadW'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=ROOT / 'weights.bin')
    parser.add_argument('--reference', type=Path, help='Write prefill and decode oracle JSON')
    args = parser.parse_args()
    torch.set_num_threads(1)
    weights = read_weights(args.weights)
    for name, expected in generate():
        if not torch.equal(weights[name], expected):
            raise ValueError(f'Seed-42 mismatch in {name}')
    digest = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    print(f'PASS: all tensors, {PARAMETERS:,} parameters, {PARAMETERS * 4:,} bytes; SHA256 {digest}')
    if args.reference:
        tokens = list('Hi 🌍'.encode('utf-8'))
        prompt = tokens.copy()
        steps = []
        with torch.no_grad():
            for _ in range(4):
                logits = reference(weights, tokens)
                token = int(logits.argmax())
                steps.append({'token': token, 'logits': logits.tolist()})
                tokens.append(token)
        args.reference.parent.mkdir(parents=True, exist_ok=True)
        args.reference.write_text(json.dumps({'prompt': prompt, 'steps': steps}) + '\n')
        print(f'Wrote independent full-prefix PyTorch oracle to {args.reference}')


if __name__ == '__main__':
    main()
