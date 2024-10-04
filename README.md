- docker build -t azure_cuda .
- docker run --env-file configs.env azure_cuda


pip list --format=freeze > requirements.txt
