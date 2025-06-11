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
    # Dynamically find files to match the CI/CD workflow
    search_dirs = ["src", "models", "deployment"]
    extensions = (".py", ".csv")
    files = []

    for d in search_dirs:
        for root, _, filenames in os.walk(d):
            for fname in filenames:
                if fname.endswith(extensions) or fname == "requirements.txt":
                    files.append(os.path.join(root, fname))

    # Add workflow file explicitly
    files.append("./.github/workflows/ci-cd.yaml")

    for file in files:
        if os.path.abspath(file) == os.path.abspath(__file__):
            print(f"Skipping conversion for {file} (script itself).")
            continue
        if os.path.exists(file):
            convert_to_utf8(file)
        else:
            print(f"{file} does not exist.")