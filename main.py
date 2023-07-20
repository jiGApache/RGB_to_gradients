import os
import cv2
import torch
import argparse
import numpy as np
import matplotlib.pylab as plt
import torchvision.transforms as t
from Models.GradientExtracion import GradientExtractor

extractor = GradientExtractor()

transform_img = t.Compose([
    t.ToTensor(),
    t.Resize((2048, 2048)),
    t.Pad(1, padding_mode='symmetric')
])

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--detect-edges', action='store_true')
args = parser.parse_args()

with torch.no_grad():
    for img_path in os.listdir('Images'):

        img = cv2.cvtColor(
            cv2.imread(
                f'Images/{img_path}'
            ),
            cv2.COLOR_BGR2RGB
        )

        # Image preprocessing
        img = np.float32(img)
        img = transform_img(img)

        horizontal, vertical = extractor(img)

        horizontal = torch.permute(horizontal, (1, 2, 0)).cpu().numpy()
        vertical = torch.permute(vertical, (1, 2, 0)).cpu().numpy()
        if args.detect_edges:
            horizontal = cv2.convertScaleAbs(horizontal)
            vertical = cv2.convertScaleAbs(vertical)
        else:
            # Values to [0, 255] and casting to uint8
            horizontal = (horizontal - np.min(horizontal)) / (np.max(horizontal) - np.min(horizontal))
            horizontal = horizontal * 255
            horizontal = horizontal.astype('uint8')
            vertical = (vertical - np.min(vertical)) / (np.max(vertical) - np.min(vertical))
            vertical = vertical * 255
            vertical = vertical.astype('uint8')

        plt.subplot(1, 2, 1)
        plt.imshow(horizontal, cmap='gray')
        plt.title('horizontal')
        plt.subplot(1, 2, 2)
        plt.imshow(vertical, cmap='gray')
        plt.title('vertical')
        plt.show()