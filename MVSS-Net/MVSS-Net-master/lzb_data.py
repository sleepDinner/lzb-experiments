from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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


def normalize_rgb(image_rgb):
    image = image_rgb.astype(np.float32) / 255.0
    image = (image - MEAN) / STD
    return image


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


class LZBSegDataset(Dataset):
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
            image = light_degradation_aug(image)
        image = normalize_rgb(image).transpose(2, 0, 1)
        mask = (mask > 0).astype(np.float32)[None, ...]
        return torch.from_numpy(image), torch.from_numpy(mask), torch.tensor(int(label > 0), dtype=torch.long), image_path


def save_prob_png(path, prob):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (np.clip(prob, 0, 1) * 255).astype("uint8"))
