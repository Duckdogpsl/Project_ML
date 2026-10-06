"""Generate labelled synthetic images solely for CI integration tests.

These are not tomato-leaf accuracy evidence and are never committed as a dataset.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def generate(output):
    rng = np.random.default_rng(42)
    colors = {'Tomato___Bacterial_spot': (180, 65, 45),
              'Tomato___Early_blight': (70, 100, 180), 'Tomato___healthy': (45, 180, 65)}
    for split in ('train', 'val', 'test'):
        for label, color in colors.items():
            folder = output / split / label
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(50):
                noise = rng.integers(-25, 26, size=(64, 64, 3))
                pixels = np.clip(np.array(color) + noise, 0, 255).astype(np.uint8)
                Image.fromarray(pixels).save(folder / f'image-{index:03}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    generate(parser.parse_args().output)
