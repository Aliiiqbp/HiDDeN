import torch
import torch.nn as nn
import numpy as np
import math
import cv2

from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor
import torchvision.transforms.functional as F_t
from random import randint, random
import skimage



class SaltPepper(nn.Module):
    def __init__(self,Amount):
        super(SaltPepper, self).__init__()
        self.Amount = Amount

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        batch_encoded_image = noised_image.cpu().detach().numpy()
        batch_encoded_image = batch_encoded_image.transpose((0, 2, 3, 1))
        for idx in range(batch_encoded_image.shape[0]):
            encoded_image = batch_encoded_image[idx]
            noise_image = skimage.util.random_noise(encoded_image, mode='s&p', amount = 0.02) #self.Amount
            noise_image = torch.from_numpy(noise_image.transpose((2, 0, 1))).type(torch.FloatTensor).cuda()
            if (idx == 0):
                batch_noise_image = noise_image.unsqueeze(0)
            else:
                batch_noise_image = torch.cat((batch_noise_image, noise_image.unsqueeze(0)), 0)  # batch*H*W*C
        batch_noise_image = Variable(batch_noise_image, requires_grad=True).cuda()  # batch*C*H*W
        noised_and_cover[0] = batch_noise_image
        return noised_and_cover

