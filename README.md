# RETFound-Green - a high performance retinal image foundation model trained with half-the-data and 400 times less compute

🚧 **This repo:** This repo contains the RETFound-Green model weights and basic instructions for usage, as well as the code for training and evaluating the model.

***If you just want to use the model, you do not need to clone this repo. Just download the weights and use the minimal snippets below in your project.***

## Basic useage
To load RETFound-Green, you need to install a recent version of pytorch (we used 2.2) and timm (we used 0.9.12), and download the model weights shared as "release" on this GitHub repo (sidebar on the right, or simply run ```!wget https://github.com/justinengelmann/RETFound_Green/releases/download/v0.1/retfoundgreen_statedict.pth```).

You can then simply load the model like this:
```python
import timm

rfg = timm.create_model('vit_small_patch14_reg4_dinov2',
                        img_size=(392, 392), num_classes=0,
                        checkpoint_path='retfoundgreen_statedict.pth').eval()
rfg.global_pool = 'avg'
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
