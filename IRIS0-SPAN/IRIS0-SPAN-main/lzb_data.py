import cv2
import keras
import numpy as np


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
    aug_index = np.random.randint(0, 8)
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


class LZBSequence(keras.utils.Sequence):
    def __init__(self, list_file, image_size=512, batch_size=1, shuffle=True, mask_size=None):
        self.pairs = read_pairs(list_file)
        self.image_size = int(image_size)
        self.mask_size = int(mask_size) if mask_size is not None else self.image_size
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.augment = bool(shuffle)
        self.indexes = np.arange(len(self.pairs))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.pairs) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, batch_index):
        batch_indexes = self.indexes[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        images = []
        masks = []
        for idx in batch_indexes:
            image_path, mask_path, _ = self.pairs[idx]
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(mask_path)
            if self.augment:
                image, mask = geometric_aug(image, mask)
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.mask_size, self.mask_size), interpolation=cv2.INTER_NEAREST)
            images.append(image.astype("float32") / 255.0 * 2.0 - 1.0)
            masks.append((mask > 0).astype("float32"))
        return np.stack(images, axis=0), np.expand_dims(np.stack(masks, axis=0), axis=-1)
