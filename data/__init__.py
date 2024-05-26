from torch.utils.data import DataLoader

from .datasets import get_dataset, BaseDataset
from .transforms import get_transform
from .collate_functions import get_collate_fn


def _get_dataset(kwargs):
    ds_transforms = get_transform(kwargs.pop("transforms"))
    name = kwargs.pop("name")
    ds = get_dataset(name=name, config=kwargs)
    ds.transform = ds_transforms
    return ds


def make_dataloader(
    dataset: BaseDataset,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
    collate_fn: str | None = None,
) -> DataLoader:
    collate_fn
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=get_collate_fn(collate_fn) if collate_fn is not None else None,
    )


def get_dataloaders(config: dict) -> tuple[DataLoader | list[DataLoader], DataLoader]:
    """Get dataloaders from config. Returns val dataloader and train dataloader(s)"""
    train_dataloaders: list[DataLoader] = []
    train = config["train"]
    for item in train:
        key, kwargs = next(iter(item.items()))
        match key:
            case "dataset":
                ds = _get_dataset(kwargs)
                dataloader = make_dataloader(
                    dataset=ds,
                    batch_size=kwargs["batch_size"],
                    num_workers=kwargs.get("num_workers", 0),
                    shuffle=kwargs.get("shuffle", True),
                    collate_fn=kwargs.get("collate_fn", None),
                )
                train_dataloaders.append(dataloader)
            case _:
                raise NotImplementedError()

    val = config["validation"]
    val_ds = _get_dataset(val)
    val_dataloader = make_dataloader(
        dataset=val_ds,
        batch_size=val["batch_size"],
        num_workers=val.get("num_workers", 0),
        shuffle=val.get("shuffle", False),
        collate_fn=val.get("collate_fn", None),
    )

    if len(train_dataloaders) == 1:
        return train_dataloaders[0], val_dataloader
    return train_dataloaders, val_dataloader
