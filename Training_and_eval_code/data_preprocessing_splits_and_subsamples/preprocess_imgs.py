import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as F


def preprocess_img(img, threshold=5, res=1024, pad_percent=0.025):
    """
    Step 1: Remove black borders
    Step 2: Pad/crop to square
    Step 3: Resize to resXres
    """
    # Step 1
    # find the left, right, top, and bottom boundaries where the image is not black
    img_array = np.array(img)
    img_array_mean = img_array.mean(-1)
    # for very dark images, scale up range
    img_array_mean = (img_array_mean / img_array_mean.max()) * 255
    # for images with a grey-ish background, we substract the background intensity
    img_array_mean -= np.quantile(img_array_mean, 0.05)
    x_filter = np.where(img_array_mean.mean(0) > threshold)[0]
    left, right = x_filter.min(), x_filter.max()
    y_filter = np.where(img_array_mean.mean(1) > threshold)[0]
    top, bottom = y_filter.min(), y_filter.max()

    # add a pad_percent% pixel buffer
    buffer = int(((img_array.shape[0] + img_array.shape[1]) / 2) * pad_percent)
    left = max(0, left - buffer)
    right = min(img_array.shape[1], right + buffer)
    top = max(0, top - buffer)
    bottom = min(img_array.shape[0], bottom + buffer)

    img = img.crop((left, top, right, bottom))

    # Step 2
    # make the image square
    width, height = img.size
    if width > height:
        # pad the top and bottom
        to_pad = width - height
        top_pad = to_pad // 2
        bottom_pad = to_pad - top_pad
        # left, top, right and bottom
        padding = [0, top_pad, 0, bottom_pad]
    else:
        # pad the left and right
        to_pad = height - width
        left_pad = to_pad // 2
        right_pad = to_pad - left_pad
        padding = [left_pad, 0, right_pad, 0]
    img = F.pad(img, padding)

    # Step 3
    img = img.resize((res, res), resample=Image.Resampling.LANCZOS)

    return img


import multiprocessing as mp
from functools import partial


def preprocess_img_mp(img_path, save_dir, threshold=5, res=1024):
    # skip if already preprocessed
    if (save_dir / img_path.name).exists():
        return

    img = Image.open(img_path)
    try:
        img = preprocess_img(img, threshold=threshold, res=res)
    except Exception as e:
        # fallback: just resize to 1024x1024
        print(f'Error processing {img_path.name}: {e}. Fallback to simple resizing...')
        img = img.resize((res, res), resample=Image.LANCZOS)
    img.save(save_dir / img_path.name, quality=90, subsampling=0)


def preprocess_imgs_mp(img_paths, save_dir, threshold=5, n_workers=mp.cpu_count() - 2):
    # use tqdm to show progress, ordering is not important
    with mp.Pool(n_workers) as pool:
        list(tqdm(pool.imap_unordered(partial(preprocess_img_mp, save_dir=save_dir, threshold=threshold), img_paths),
                  total=len(img_paths)))


if __name__ == '__main__':
    img_paths = Path(r"E:\retinal_images\datasets\ODIR").rglob('*.jpg')
    img_paths = list(img_paths)
    save_dir = Path(r"E:\retinal_images\all\ODIR")

    print(f'Preprocessing {len(img_paths)} images...')

    preprocess_imgs_mp(img_paths, save_dir)
