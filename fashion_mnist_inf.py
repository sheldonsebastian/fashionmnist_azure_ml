# %%
import torch
import os
import gzip
import numpy as np
import torch.nn as nn

# %%
SAVED_MODEL_PATH = r"fashion_mnist\saved_models\fashion_mnist_model.pth"


# %%
def load_data(root, train=True):
    if train:
        images_path = os.path.join(root, "train-images-idx3-ubyte.gz")
        labels_path = os.path.join(root, "train-labels-idx1-ubyte.gz")
    else:
        images_path = os.path.join(root, "t10k-images-idx3-ubyte.gz")
        labels_path = os.path.join(root, "t10k-labels-idx1-ubyte.gz")

    with gzip.open(labels_path, "rb") as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, "rb") as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(
            len(labels), 28, 28
        )

    return images, labels


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
# read saved model checkpoints
model = SimpleNN()
model.load_state_dict(torch.load(SAVED_MODEL_PATH))
model.eval()

# %%
# Load Fashion MNIST dataset
image, label = load_data(r"fashion_mnist\data\FashionMNIST\raw", train=False)

# %% make predictions
# Convert images to tensor
images_tensor = torch.tensor(image, dtype=torch.float32) / 255.0

# Get predictions
with torch.no_grad():
    outputs = model(images_tensor)
    _, predicted = torch.max(outputs, 1)

# Print predictions for the first 10 images
for i in range(10):
    print(
        f"Image {i+1}: Predicted label: {predicted[i].item()}, True label: {label[i]}"
    )

# %%
print("Finito")
