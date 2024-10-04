# Import necessary libraries
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.operations import DatastoreOperations
from dotenv import load_dotenv
import os

# Load environment variables
if os.path.exists("configs.env"):
    load_dotenv("configs.env")

# Authenticate with Azure
credential = DefaultAzureCredential()

# Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id=os.getenv("SUBSCRIPTION_ID"),
    resource_group_name=os.getenv("RESOURCE_GROUP"),
    workspace_name=os.getenv("WORKSPACE_NAME"),
)

# Define the path to the local data
data_path = "data"

# Set the version number of the data asset
v1 = "initial"

# Define the data asset
my_data = Data(
    name="fashion-mnist",
    version=v1,
    description="Fashion MNIST dataset",
    path=data_path,
    type=AssetTypes.URI_FOLDER,
)

# Create the data asset if it doesn't exist
try:
    data_asset = ml_client.data.get(name="fashion-mnist", version=v1)
    print(
        f"Data asset already exists. Name: {my_data.name}, version: {my_data.version}"
    )
except Exception as e:
    ml_client.data.create_or_update(my_data)
    print(f"Data asset created. Name: {my_data.name}, version: {my_data.version}")

# Retrieve the data asset
data_asset = ml_client.data.get(name="fashion-mnist", version=v1)
print(f"Data asset: {data_asset.name}, version: {data_asset.version}")

# Download the data asset using the MLClient's data download method
local_directory = "downloaded_data"
os.makedirs(local_directory, exist_ok=True)

# Download the asset to the specified local directory
ml_client.data.download(name="fashion-mnist", version=v1, download_path=local_directory)

print(f"Data downloaded to {local_directory}")
