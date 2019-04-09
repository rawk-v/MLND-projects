import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

is_debug = False


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(4, stride=4)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # 128*28*28
        self.pool3 = nn.MaxPool2d(2)  # 128*7*7
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 7 * 7, 136)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        if is_debug:
            print('pool4.out_shape:', x.shape)
        x = x.view(x.size(0), -1)
        x = self.drop1(x)
        x = self.fc1(x)
        if is_debug:
            print('fc1.out_shape:', x.shape)
        # a modified x, having gone through all the layers of your model, should be returned
        return x


if __name__ == '__main__':
    import os
    os.chdir('/Users/rawk/Projects/Project Zero/MLND-projects/cv-project0-facial-keypoints/')

    from data_load import FacialKeypointsDataset
    from data_load import Rescale, RandomCrop, Normalize, ToTensor
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, utils

    data_transform = transforms.Compose([Rescale((250, 250)), RandomCrop((224, 224)), Normalize(), ToTensor()])

    transformed_dataset = FacialKeypointsDataset(csv_file='./data/training_frames_keypoints.csv',
                                                 root_dir='./data/training/',
                                                 transform=data_transform)
    batch_size = 2

    data_loader = DataLoader(transformed_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=4)

    for i, sample in enumerate(data_loader):
        images = sample['image']
        images = images.type(torch.FloatTensor)
        net = Net()
        net(images)
        if i == 0:
            break


