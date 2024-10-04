FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.2-cuda12.1:18

COPY environment.yml .
RUN conda env create -f environment.yml
