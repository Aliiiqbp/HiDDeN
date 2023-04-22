import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from noise_layers.identity import Identity
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.quantization import Quantization
from noise_layers.gaussian import Gaussian
from noise_layers.gaussianblur import GaussianBlur
from noise_layers.saltpepper import SaltPepper


def convert_to_comp(imgs, comp_type = "Compression"):
        comp_imgs = torch.zeros((imgs.shape))
        
        for i in range(imgs.shape[0]):
            imgs[i] = imgs[i]*0.5+0.5
            pil_img = transforms.ToPILImage()(imgs[i])
            # pil_img = imgs[i]

            
            if comp_type == "WebP":
                pil_img.save(str(i+1) + ".webp", format="webp")
                comp_img = Image.open(str(i+1) + ".webp").convert('RGB')
            elif comp_type == "JPEG2000":
                pil_img.save(str(i+1) + ".jp2", format = "JPEG2000")
                comp_img = Image.open(str(i+1)+".jp2").convert('RGB')
            elif comp_type == "Compression":
                pil_img.save(str(i+1) + ".jpg", format = "JPEG", quality=85)
                comp_img = Image.open(str(i+1)+".jpg").convert('RGB')

            comp_img = transforms.PILToTensor()(comp_img).float()
            comp_img = transforms.CenterCrop(128)(comp_img)
            comp_img = comp_img/255.0
            comp_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(comp_img)
            
            comp_imgs[i] = comp_img
        return comp_imgs




class Noiser(nn.Module):
    """
    This module allows to combine different noise layers into a sequential noise module. The
    configuration and the sequence of the noise layers is controlled by the noise_config parameter.
    """
    def __init__(self, noise_layers: list, device):
        super(Noiser, self).__init__()
        self.device = device
        self.noise_layers = [Identity()]
        for layer in noise_layers:
            if type(layer) is str:
                if layer == 'JpegPlaceholder':
                    self.noise_layers.append(JpegCompression(device))
                elif layer == 'QuantizationPlaceholder':
                    self.noise_layers.append(Quantization(device))
                elif layer == 'Gaussian':
                    self.noise_layers.append(Gaussian(device))
				#elif layer == 'SaltPepper':
				#	self.noise_layers.append(SaltPepper(device))
                elif layer == 'GaussianBlur':
                    self.noise_layers.append(GaussianBlur(device))
                else:
                    raise ValueError(f'Wrong layer placeholder string in Noiser.__init__().'
                                     f' Expected "JpegPlaceholder" or "QuantizationPlaceholder" but got {layer} instead')
            else:
                self.noise_layers.append(layer)
        # self.noise_layers = nn.Sequential(*noise_layers)
#        self.noise_layers = [SaltPepper(device)]


    def forward(self, encoded_and_cover):
#        encoded_and_cover[0] = convert_to_comp(encoded_and_cover[0], comp_type = "Compression").to(self.device)
#        encoded_and_cover[1] = convert_to_comp(encoded_and_cover[1], comp_type = "Compression").to(self.device)
#        print("real jpeg")

        random_noise_layer = np.random.choice(self.noise_layers, 1)[0]
        print(random_noise_layer)
        #print(self.noise_layers[5])
        #return self.noise_layers[5](encoded_and_cover)
        return random_noise_layer(encoded_and_cover)
#        return encoded_and_cover

