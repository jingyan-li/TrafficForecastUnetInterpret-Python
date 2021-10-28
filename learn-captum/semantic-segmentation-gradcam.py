#  Author: 2021. Jingyan Li

# Semantic Segmentation with Captum
# Link: https://captum.ai/tutorials/Segmentation_Interpret

from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np

from torchvision import models
from torchvision import transforms

from captum.attr import visualization as viz
from captum.attr import LayerGradCam, FeatureAblation, LayerActivation, LayerAttribution

# Default device
device = "cuda" if torch.cuda.is_available() else "cpu"

#%%
# load the pre-trained segmentation model from torchvision, which is trained on a subset of COCO Train 2017 and define input preprocessing transforms.
fcn = models.segmentation.fcn_resnet101(pretrained=True).to(device).eval()

# Input preprocessing transformation
preprocessing = transforms.Compose([transforms.Resize(640),
                                    transforms.ToTensor()])
normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

# This method allows us to visualize a particular segmentation output, by setting
# each pixels color according to the given segmentation class provided in the
# image (segmentation output).
def decode_segmap(image, nc=21):
    label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

#%%
# Display original image
img = Image.open("data/8862358875_eecba9fb10_z.jpg")
preproc_img = preprocessing(img)
plt.imshow(preproc_img.permute(1,2,0)); plt.axis('off'); plt.show()

#%%
# Normalize image and compute segmentation output
normalized_inp = normalize(preproc_img).unsqueeze(0).to(device)
normalized_inp.requires_grad = True
out = fcn(normalized_inp)['out']

#%%
# Find most likely segmentation class for each pixel.
out_max = torch.argmax(out, dim=1, keepdim=True)

# Visualize segmentation output using utility method.
rgb = decode_segmap(out_max.detach().cpu().squeeze().numpy())
plt.imshow(rgb); plt.axis('off'); plt.show()

#%%
# def agg_segmentation_wrapper(inp):
#     model_out = fcn(inp)['out']
#     # Creates binary matrix with 1 for original argmax class for each pixel
#     # and 0 otherwise. Note that this may change when the input is ablated
#     # so we use the original argmax predicted above, out_max.
#     selected_inds = torch.zeros_like(model_out[0:1]).scatter_(1, out_max, 1)
#     return (model_out * selected_inds).sum(dim=(2,3))
#
# lgc = LayerGradCam(agg_segmentation_wrapper, fcn.backbone.layer4[2].conv3)
# gc_attr = lgc.attribute(normalized_inp, target=6)

#%%
from captum.attr import Saliency
sa = Saliency(fcn)
normalized_inp.requires_grad = True
attr = sa.attribute(normalized_inp, target=11)
#%%
attr = attr.detach().numpy()
#%%
