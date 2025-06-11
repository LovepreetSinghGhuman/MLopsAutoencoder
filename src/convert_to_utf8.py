import os
import chardet

def convert_to_utf8(filename):
    with open(filename, 'rb') as f:
        raw = f.read()
        detected = chardet.detect(raw)
        encoding = detected['encoding']

    if encoding is None:
        print(f"Could not detect encoding for {filename}. Skipping.")
        return

    if encoding.lower() == 'utf-8':
        print(f"{filename} is already UTF-8 encoded.")
        return

    with open(filename, 'r', encoding=encoding, errors='replace') as f:
        content = f.read()
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Converted {filename} from {encoding} to UTF-8.")

if __name__ == "__main__":
    files = [
        "./data/processed/cleaned_test.csv",
        "./data/processed/cleaned_train.csv",
        "./deployment/conda.yaml",
        "./deployment/Dockerfile",
        "./deployment/k8s/deployment.yaml",
        "./deployment/k8s/ingress.yaml",
        "./deployment/k8s/service.yaml",
        "./deployment/train-job.yaml",
        "./frontend/frontend-nginx-deployment.yaml",
        "./frontend/frontend-nginx.yaml",
        "./frontend/index.html",
        "./.github/workflows/ci-cd.yaml",  # Ensure workflow file is UTF-8
    ]
    for file in files:
        if os.path.abspath(file) == os.path.abspath(__file__):
            print(f"Skipping conversion for {file} (script itself).")
            continue
        if os.path.exists(file):
            convert_to_utf8(file)
        else:
            print(f"{file} does not exist.")