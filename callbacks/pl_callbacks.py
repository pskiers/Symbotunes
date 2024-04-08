from pytorch_lightning import callbacks


def get_pl_callback(name: str, kwargs: dict) -> callbacks.Callback:
    try:
        cls_object = getattr(callbacks, name)
    except AttributeError:
        raise AttributeError(f"No {name} callback found in pytorch_lightning")
    return cls_object(**kwargs)
