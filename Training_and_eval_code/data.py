import numpy as np
import torch
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as TF
from torchvision.io import read_image
from torchvision.tv_tensors import Image
from pathlib import Path

DEFAULT_MEAN_STD = ([0.5], [0.5])


class RandomSidePadding(T.Transform):
    # Adding black space to the left and right of the image, to simulate bad cropping
    # we pre-processed the fundus images to remove the black borders already,
    # but we want to be robust in case users forget to do this
    def __init__(self, padding: tuple[int, int]):
        self.padding = padding

    def __call__(self, img: Image) -> Image:
        pad_amt = np.random.randint(*self.padding)
        return TF.pad(img, (pad_amt, 0, pad_amt, 0), fill=0)


def get_default_transforms(res=392):
    ts = [T.Resize((res, res), antialias=True),
          T.ToDtype(torch.float32, scale=True),
          T.Normalize(*DEFAULT_MEAN_STD)]
    return T.Compose(ts)


def get_default_untransform():
    return T.Compose([T.Normalize(mean=[-DEFAULT_MEAN_STD[0] / DEFAULT_MEAN_STD[1]], std=[1 / DEFAULT_MEAN_STD[1]])])


def get_default_training_transforms(res=392):
    cjitter = T.RandomApply([T.ColorJitter(brightness=0.33, contrast=0.33)], p=0.25)
    slight_rotation = T.RandomApply([T.RandomRotation(degrees=12.5)], p=0.25)
    side_padding = T.RandomApply([RandomSidePadding((33, 150))], p=0.1)
    scale = T.RandomApply([T.RandomAffine(degrees=0, scale=(0.8, 1.2))], p=1.)

    random_crop = T.RandomResizedCrop(res, scale=(0.7, 1.0), ratio=(3.0 / 5.0, 5.0 / 3.0), antialias=True)
    center_crop = T.Compose([T.Resize(int(res * 1.3), antialias=True), T.CenterCrop(res)])
    simple_resize = T.Resize((res, res), antialias=True)

    resize = T.RandomChoice([random_crop, center_crop, simple_resize])

    ts = [T.RandomHorizontalFlip(),
          T.RandomVerticalFlip(p=0.05),
          side_padding,
          slight_rotation,
          cjitter,
          resize,
          T.ToDtype(torch.float32, scale=True),
          T.Normalize(*DEFAULT_MEAN_STD)
          ]
    return T.Compose(ts)


class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir: str | Path | list[Path], transform: torch.nn.Module = None):
        self.img_dir = img_dir
        if isinstance(img_dir, (str, Path)):
            self.img_paths = list(Path(img_dir).rglob('*.jpg'))
        else:
            self.img_paths = img_dir
        self.transform = transform or get_default_training_transforms()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = read_image(str(img_path))
        img = self.transform(img)
        return img


def get_dataloader(dataset, batch_size=32, num_workers=4, shuffle=True, pin_memory=True, prefetch_factor=2,
                   persistent_workers=True):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory,
                                       prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)
