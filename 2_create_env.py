# https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-environments-v2?view=azureml-api-2&tabs=python
"""Create docker environment for training the model"""
# %%
# import required libraries
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
import os

# %%
if os.path.exists("configs.env"):
    load_dotenv("configs.env")

# %%
# Enter details of your AML workspace
subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RESOURCE_GROUP")
workspace = os.getenv("WORKSPACE_NAME")

# %%
# get a handle to the workspace
ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
)

# %%
# create custom environment on top of Azure Custom PyTorch (ACPT) environment
# https://learn.microsoft.com/en-us/azure/machine-learning/how-to-azure-container-for-pytorch-environment?view=azureml-api-2
env_docker_conda = Environment(
    image="mcr.microsoft.com/azureml/curated/acpt-pytorch-2.2-cuda12.1:18",
    conda_file="environment.yml",
    name="custom-acpt-env",
    description="Environment created from a Base ACPT Docker image plus Conda environment for FashionMNIST.",
)

# %%
ml_client.environments.create_or_update(env_docker_conda)

# %%
envs = ml_client.environments.list()
for env in envs:
    print(env.name)
