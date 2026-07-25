import torch
from torch.utils.data import DataLoader

import dataset
from config import (
    PREPROCESSED_DIR,
    BATCH_SIZE,
)


def data_load():
    loaded = {}
    
    for name in ["train", "valid", "test"]:
        load_path = PREPROCESSED_DIR / f"{name}_encodings.pt"
        encodings = torch.load(load_path)
        
        load_path = PREPROCESSED_DIR / f"{name}_label.pt"
        label = torch.load(load_path)
        
        loaded[name] = {
            "encodings": encodings,
            "label": label
        }

    return loaded


def get_loader():
    data = data_load()

    loaders = {}

    for name in ["train", "valid", "test"]:
        ds = dataset.BertDataset(
            data[name]["encodings"],
            data[name]["label"],
        )

        loaders[name] = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=(name == "train"),
        )

    return loaders["train"], loaders["valid"], loaders["test"]