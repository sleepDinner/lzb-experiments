import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def read_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            pairs.append((parts[0], parts[1], int(float(parts[2])) if len(parts) > 2 else 1))
    return pairs


def geometric_aug(image, mask):
    aug_index = random.randrange(8)
    if aug_index == 1:
        image = np.rot90(image, 1)
        mask = np.rot90(mask, 1)
    elif aug_index == 2:
        image = np.rot90(image, 2)
        mask = np.rot90(mask, 2)
    elif aug_index == 3:
        image = np.rot90(image, 3)
        mask = np.rot90(mask, 3)
    elif aug_index == 4:
        image = np.flipud(image)
        mask = np.flipud(mask)
    elif aug_index == 5:
        image = np.flipud(np.rot90(image, 1))
        mask = np.flipud(np.rot90(mask, 1))
    elif aug_index == 6:
        image = np.flipud(np.rot90(image, 2))
        mask = np.flipud(np.rot90(mask, 2))
    elif aug_index == 7:
        image = np.flipud(np.rot90(image, 3))
        mask = np.flipud(np.rot90(mask, 3))
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def light_degradation_aug(image_rgb):
    image = np.ascontiguousarray(image_rgb)
    aug_choice = float(np.random.random())
    if aug_choice < 0.12:
        sigma = float(np.random.uniform(0.2, 0.8))
        image = cv2.GaussianBlur(image, (3, 3), sigmaX=sigma)
    elif aug_choice < 0.27:
        sigma = float(np.random.uniform(1.0, 3.0))
        noise = np.random.normal(0.0, sigma, image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    elif aug_choice < 0.40:
        quality = int(np.random.randint(85, 101))
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, enc = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            image = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(image)


class LZBManTraDataset(Dataset):
    def __init__(self, list_file, image_size=512, train=False):
        self.pairs = read_pairs(list_file)
        self.image_size = int(image_size)
        self.train = train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path, label = self.pairs[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        if self.train:
            image, mask = geometric_aug(image, mask)
            image = light_degradation_aug(image)

        image = torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1))
        mask = torch.from_numpy((mask > 0).astype(np.float32))[None, ...]
        return image, mask, torch.tensor(int(label > 0), dtype=torch.long), image_path


def save_prob_png(path, prob):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (np.clip(prob, 0, 1) * 255).astype("uint8"))
