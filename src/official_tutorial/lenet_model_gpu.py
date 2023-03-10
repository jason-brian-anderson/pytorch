import sys

batch = 8
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

sys.path.insert(1, '/workspaces/pytorch_work')
import src.lib.utils as utils

device = utils.get_device()


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 200)
        self.fc2 = nn.Linear(200, 800)
        self.fc3 = nn.Linear(800, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # print(f"x:::{x.shape}")
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        """
        asdf
        """
        size = x.size()[1:]  # all dimensions except batch dimension

        num_features = 1
        for s in size:
            num_features *= s
        # print("n:::", num_features)
        return num_features


net = LeNet()
net = net.to(device)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        ),
    ]
)


trainset = torchvision.datasets.CIFAR10(
    root="./data/train/",
    train=True,
    download=True,
    transform=transform,
)

testset = torchvision.datasets.CIFAR10(
    root="./data/test/",
    train=False,
    download=True,
    transform=transform,
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch,
    shuffle=True,
    num_workers=1,
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=batch,
    shuffle=True,
    num_workers=1,
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    net.parameters(),
    lr=0.001,
    momentum=0.9,
)




epochs = 3

for epoch in range(epochs):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % batch * 10000 == 0:
            print(f"[{i+1}], epoch: {epoch}: {10000 * running_loss / 2000:.3f}")
        running_loss = 0.0


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()


print(correct / total * 100, "%")
