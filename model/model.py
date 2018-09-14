from base import BaseModel
import torch.nn as nn
import torch.nn.functional as F


class seabedModel(BaseModel):
    def __init__(self, config):
        super(MnistModel, self).__init__(config)
        self.config = config
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        
        self.conv1 = nn.Conv2d(1, 10, kernel_size= self.kernel1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size= self.kernel1)
        self.conv2_drop = nn.Dropout2d()

        self.conv3 = nn.Conv2d(20, 40, kernel_size= self.kernel2)
        self.conv4 = nn.Conv2d(40, 80, kernel_size= self.kernel2)
        self.conv4_drop = nn.Dropout2d()

        self.conv5 = nn.Conv2d(80, 160, kernel_size= self.kernel1)
        self.conv6 = nn.Conv2d(160, 320, kernel_size= self.kernel1)
        self.conv6_drop = nn.Dropout2d()

        self.conv7 = nn.Conv2d(320, 640, kernel_size = self.kernel1)
        self.conv8 = nn.Conv2d(640, 800, kernel_size = self.kernel2)
        self.conv8_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv4_drop(self.conv4(x)), 2))
        x = F.relu(F.max_pool2d(self.conv5(x), 2))
        x = F.relu(F.max_pool2d(self.conv6_drop(self.conv6(x)), 2))
        x = F.relu(F.max_pool2d(self.conv7(x), 2))
        x = F.relu(F.max_pool2d(self.conv8_drop(self.conv8(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


