import os
import tempfile

import numpy as np
from PIL import Image

from Splicing.data.AbstractDataset import AbstractDataset


class LZBPairDataset(AbstractDataset):
    """Generic absolute-path image/mask list for the LZB experiments."""

    def __init__(self, crop_size, grid_crop, blocks, dct_channels, list_file, read_from_jpeg=True, resize_to=None):
        super().__init__(crop_size, grid_crop, blocks, dct_channels)
        self.read_from_jpeg = read_from_jpeg
        self.resize_to = tuple(resize_to) if resize_to is not None else None
        self.tamp_list = []
        with open(list_file, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                parts = line.split("\t") if "\t" in line else line.split()
                if len(parts) < 2:
                    continue
                self.tamp_list.append((parts[0], parts[1]))
        if not self.tamp_list:
            raise RuntimeError("Empty LZB list: {}".format(list_file))

    def _as_jpeg(self, image_path):
        if self.resize_to is None and (not self.read_from_jpeg or image_path.lower().endswith((".jpg", ".jpeg"))):
            return image_path, None
        fd, temp_path = tempfile.mkstemp(prefix="cat_lzb_", suffix=".jpg")
        os.close(fd)
        image = Image.open(image_path).convert("RGB")
        if self.resize_to is not None and image.size != self.resize_to:
            image = image.resize(self.resize_to, Image.BILINEAR)
        image.save(temp_path, quality=100, subsampling=0)
        return temp_path, temp_path

    def _read_mask_checked(self, mask_path, image_path):
        mask_image = Image.open(mask_path).convert("L")
        with Image.open(image_path) as image:
            image_size = image.convert("RGB").size
        if mask_image.size != image_size:
            raise ValueError(
                "CAT-Net LZB list contains mismatched image/mask sizes. "
                "Rebuild lists with the strict pair filter. "
                "image={} image_size={} mask={} mask_size={}".format(
                    image_path, image_size, mask_path, mask_image.size
                )
            )
        mask = np.array(mask_image)
        mask[mask > 0] = 1
        return mask

    def _read_mask_for_original_or_resized_image(self, mask_path, original_image_path, jpeg_path):
        if self.resize_to is None:
            return self._read_mask_checked(mask_path, jpeg_path)
        mask_image = Image.open(mask_path).convert("L")
        with Image.open(original_image_path) as image:
            original_size = image.convert("RGB").size
        if mask_image.size != original_size:
            raise ValueError(
                "CAT-Net LZB list contains mismatched image/mask sizes. "
                "Rebuild lists with the strict pair filter. "
                "image={} image_size={} mask={} mask_size={}".format(
                    original_image_path, original_size, mask_path, mask_image.size
                )
            )
        if mask_image.size != self.resize_to:
            mask_image = mask_image.resize(self.resize_to, Image.NEAREST)
        mask = np.array(mask_image)
        mask[mask > 0] = 1
        return mask

    def get_tamp(self, index):
        image_path, mask_path = self.tamp_list[index]
        jpeg_path, temp_path = self._as_jpeg(image_path)
        try:
            mask = self._read_mask_for_original_or_resized_image(mask_path, image_path, jpeg_path)
            return self._create_tensor(jpeg_path, mask)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def __getitem__(self, index):
        return self.get_tamp(index)
