# Read input and save output: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-read-write-data-v2?view=azureml-api-2&tabs=python
# Save metrics and models: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-log-view-metrics?view=azureml-api-2&tabs=jobs
# Quickstart Notebook: https://github.com/Azure/azureml-examples/blob/main/tutorials/get-started-notebooks/quickstart.ipynb
import os
import gzip
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import VisionDataset
from azureml.fsspec import AzureMachineLearningFileSystem
from dotenv import load_dotenv
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import mlflow

# Start Logging
mlflow.start_run()

# enable autologging
mlflow.autolog()


if os.path.exists("configs.env"):
    load_dotenv("configs.env")


# authenticate
credential = DefaultAzureCredential()


# Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id=os.getenv("SUBSCRIPTION_ID"),
    resource_group_name=os.getenv("RESOURCE_GROUP"),
    workspace_name=os.getenv("WORKSPACE_NAME"),
)

print("GPU found:", torch.cuda.is_available())


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


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)


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


# download the dataset
def download_data(ml_client):
    data_asset = ml_client.data.get(name="fashion-mnist", version="latest")

    # instantiate file system using following URI
    fs = AzureMachineLearningFileSystem(data_asset.path)

    # list folders/files in datastore 'datastorename'
    found_files = fs.ls()

    # download all files into data2 folder
    DATA_ROOT_DIR = "data"
    if not os.path.exists(DATA_ROOT_DIR):
        for file in found_files:
            fs.download(file, DATA_ROOT_DIR)
    return DATA_ROOT_DIR


DATA_ROOT_DIR = download_data(ml_client)

# Load dataset from ubyte and ubyte.gz files
train_dataset = FashionMNISTFromUbyte(
    root=DATA_ROOT_DIR,
    transform=transform,
)

mlflow.log_metric("num_samples", len(train_dataset))

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=64, shuffle=True
)

# Initialize the network, loss function and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

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

    mlflow.log_metric("loss", loss.item(), step=epoch)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training complete.")


SAVE_PATH = "./output"
os.makedirs(SAVE_PATH, exist_ok=True)
model_save_path = os.path.join(SAVE_PATH, "fashion_mnist_model.pth")
torch.save(model.state_dict(), model_save_path)

# Log the model with mlflow
mlflow.pytorch.log_model(model, "fashion_mnist_model")

# Registering the model to the workspace
print("Registering the model via MLFlow")
mlflow.register_model(
    model_uri="runs:/{}/fashion_mnist_model".format(mlflow.active_run().info.run_id),
    name="fashion_mnist_model",
)

print(f"Model saved in {model_save_path}")
