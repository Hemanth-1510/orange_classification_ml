import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
from skimage.color import rgb2hsv, rgb2lab, rgb2gray

def extract_features(image_path):
    rgb = np.asarray(Image.open(image_path).convert("RGB").resize((128, 128)))
    parts = []
    for space in (rgb2hsv(rgb), rgb2lab(rgb)):
        for ch in range(3):
            h, _ = np.histogram(space[:, :, ch].ravel(), bins=32)
            h = h.astype(np.float32); h /= max(h.sum(), 1.0); parts.append(h)
    lbp = local_binary_pattern(rgb2gray(rgb), 24, 3, method="uniform")
    h, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 27))
    h = h.astype(np.float32); h /= max(h.sum(), 1.0); parts.append(h)
    return np.concatenate(parts).astype(np.float32)
