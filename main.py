import kagglehub

# Download latest version
path = kagglehub.dataset_download("henryshan/google-stock-price")

print("Path to dataset files:", path)