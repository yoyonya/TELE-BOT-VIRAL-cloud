import os

def load_documents(folder="data"):
    docs = []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                docs.append({
                    "text": f.read(),
                    "source": file
                })
    return docs
