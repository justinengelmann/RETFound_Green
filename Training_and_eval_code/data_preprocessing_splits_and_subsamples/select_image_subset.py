import numpy as np
from pathlib import Path

imgs_ddr = list(Path('/datastorage/justin/retinal_images/DDR').rglob('*.jpg'))
imgs_odir = list(Path('/datastorage/justin/retinal_images/ODIR').rglob('*.jpg'))
imgs_airogs = list(Path('/datastorage/justin/retinal_images/AIROGS').rglob('*.jpg'))

# we want exactly 75,000 images for training
n_airogs_to_use = 75_000 - (len(imgs_ddr) + len(imgs_odir))

np.random.seed(42)
imgs_airogs_select = np.random.choice(imgs_airogs, size=n_airogs_to_use, replace=False)
print(len(imgs_airogs_select))
np.save('imgs_airogs_select.npy', imgs_airogs_select)
imgs_all = list(imgs_airogs_select) + imgs_ddr + imgs_odir
print(len(imgs_all))
np.save('imgs_all.npy', imgs_all)
