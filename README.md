# RETFound-Green - a high performance retinal image foundation model trained with half-the-data and 400 times less compute

🚧 **Under construction:** At present, this repo contains the RETFound-Green model weights and basic instructions for usage. We're currently cleaning up and documenting the code for training and evaluation. Email us if you encounter any issues or would like a preview version.

## Basic useage
To load RETFound-Green, you need to install a recent version of pytorch (we used 2.2) and timm (we used 0.9.12), and download the model weights shared as "release" on this GitHub repo (sidebar on the right, or click this [link](https://github.com/justinengelmann/RETFound_Green/releases/tag/v0.1)).

You can then simply load the model like this:
```python
import timm
import torch

rfg = timm.create_model('vit_small_patch14_reg4_dinov2',
                        img_size=(392, 392), num_classes=0).eval()
rfg_weights = torch.load('retfoundgreen_statedict.pth')
rfg.load_state_dict(rfg_weights)
```

No additional custom code is needed. You do not need to clone this GitHub repo to use RETFound-Green.

Note that the model uses normalisation constants of 0.5 for all three channel means and standard deviations. Please use the modern torchvision "v2" API and pytorch's read_image function. Your data transformation and inference should look something like this:

```python
import torch
from torchvision.transforms import v2 as T
from torchvision.io import read_image

transforms = T.Compose([T.Resize((392, 392), antialias=True),
                        T.ToDtype(torch.float32, scale=True),
                        T.Normalize((0.5,), (0.5,))])

img = read_image('path/to/your/img.jpg')
img = transforms(img)
# add dummy batch dimension
img = img.unsqueeze(0)
with torch.inference_mode():
    # get model features and remove dummy batch dimension
    features = rfg(img).squeeze()
```

For faster inference, use a pytorch dataloader for batch inference and use GPU-acceleration if you have a recent nvidia GPU. However, processing images one-by-one on CPU is reasonably fast, too, in our experience.
