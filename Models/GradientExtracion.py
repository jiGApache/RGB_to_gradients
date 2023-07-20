import torch
import torch.nn as nn

class GradientExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_rgb2gray = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(1,1))
        # RGB to Grayscale using Relative Luminanve defenition
        self.conv_rgb2gray.weight.data = \
            torch.as_tensor([[[[0.2126]], [[0.7152]], [[0.0722]]]], dtype=torch.float32)
        self.conv_rgb2gray.bias.data.fill_(0.0)

        self.conv_horizontal_grad = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3))
        self.conv_horizontal_grad.weight.data = \
            torch.as_tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)
        self.conv_horizontal_grad.bias.data.fill_(0.0)
        
        self.conv_vertical_grad = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3))
        self.conv_vertical_grad.weight.data = \
            torch.as_tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
        self.conv_vertical_grad.bias.data.fill_(0.0)

    def forward(self, img):
        img = self.conv_rgb2gray(img)
        return self.conv_horizontal_grad(img), self.conv_vertical_grad(img)