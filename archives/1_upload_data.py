# https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-explore-data?view=azureml-api-2
"""Upload data to Azure ML workspace."""

# %%
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from dotenv import load_dotenv
import os


# %%
if os.path.exists("configs.env"):
    load_dotenv("configs.env")

# %%
# authenticate
credential = DefaultAzureCredential()

# %%
# Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id=os.getenv("SUBSCRIPTION_ID"),
    resource_group_name=os.getenv("RESOURCE_GROUP"),
    workspace_name=os.getenv("WORKSPACE_NAME"),
)

# %%
data_path = "data"

# %%
# set the version number of the data asset
v1 = "v2"

# %%
my_data = Data(
    name="fashion-mnist",
    version=v1,
    description="Fashion MNIST dataset",
    path=data_path,
    type=AssetTypes.URI_FOLDER,
)

# %%
## create data asset if it doesn't already exist:
try:
    data_asset = ml_client.data.get(name="fashion-mnist", version=v1)
    print(
        f"Data asset already exists. Name: {my_data.name}, version: {my_data.version}"
    )
except:
    ml_client.data.create_or_update(my_data)
    print(f"Data asset created. Name: {my_data.name}, version: {my_data.version}")

# %%
# consuming the data
data_asset = ml_client.data.get(name="fashion-mnist", version=v1)
print(f"Data asset: {data_asset.name}, version: {data_asset.version}")

# %%
# https://learn.microsoft.com/en-us/azure/machine-learning/how-to-access-data-interactive?view=azureml-api-2&tabs=datastore
datastore_path = data_asset.path

# %%
from azureml.fsspec import AzureMachineLearningFileSystem

# %%
# instantiate file system using following URI
fs = AzureMachineLearningFileSystem(datastore_path)

# %%
# list folders/files in datastore 'datastorename'
found_files = fs.ls()

# %%
print(found_files)

# %%
# download all files into data2 folder
for file in found_files:
    fs.download(file, f"downloaded_data")

# %%
