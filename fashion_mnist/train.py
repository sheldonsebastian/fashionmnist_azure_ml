# %%
import torch
from torchvision import transforms
import os
import gzip
import numpy as np
from torchvision.datasets import VisionDataset

import torch.nn as nn
import torch.optim as optim

# %%
print("GPU found:", torch.cuda.is_available())

# %%
# Change directory to the current file directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# %%
# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# %%
# Load Fashion MNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)


# %%
class FashionMNISTFromUbyte(VisionDataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(FashionMNISTFromUbyte, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        self.train = train
        self.data, self.targets = self.load_data()

    def load_data(self):
        if self.train:
            images_path = os.path.join(self.root, "train-images-idx3-ubyte.gz")
            labels_path = os.path.join(self.root, "train-labels-idx1-ubyte.gz")
        else:
            images_path = os.path.join(self.root, "t10k-images-idx3-ubyte.gz")
            labels_path = os.path.join(self.root, "t10k-labels-idx1-ubyte.gz")

        with gzip.open(labels_path, "rb") as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(images_path, "rb") as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(
                len(labels), 28, 28
            )

        return images, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


# %%
# Load dataset from ubyte and ubyte.gz files
train_dataset = FashionMNISTFromUbyte(
    root=r"data",
    transform=transform,
)

# %%
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=64, shuffle=True
)

# %%
# Initialize the network, loss function and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# %%
# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# %%
print("Training complete.")

# %%
# Save the model
SAVE_PATH = "saved_models"
os.makedirs(SAVE_PATH, exist_ok=True)
torch.save(model.state_dict(), os.path.join(SAVE_PATH, "fashion_mnist_model.pth"))

# %%
