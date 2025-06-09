from torch.utils.data import Dataset
from .constants import CLASSES, DATA_PACK
import numpy as np
import os

class HandDrawDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return len(CLASSES) * DATA_PACK

    def __getitem__(self, idx):
        idx_label = int(idx / DATA_PACK)
        file_path = os.path.join("data", "full_numpy_bitmap_{}.npy".format(list(CLASSES.keys())[idx_label]))
        idx_images = np.load(file_path)[idx % DATA_PACK].astype(np.float32)
        idx_images /= 255
        return idx_images.reshape((1, 28, 28)), idx_label
