import os

RAW_DIR = "data/raw"
CHUNK_DIR = "data/chunks"
CHUNK_SIZE = 500  # Можно регулировать размер

os.makedirs(CHUNK_DIR, exist_ok=True)

def chunk_text(text, size=CHUNK_SIZE):
    return [text[i:i+size] for i in range(0, len(text), size)]

def process_files():
    for filename in os.listdir(RAW_DIR):
        if not filename.endswith(".txt"):
            continue

        with open(os.path.join(RAW_DIR, filename), "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        base_name = filename.replace(".txt", "")
        for i, chunk in enumerate(chunks):
            chunk_filename = f"{base_name}_chunk_{i+1}.txt"
            with open(os.path.join(CHUNK_DIR, chunk_filename), "w", encoding="utf-8") as cf:
                cf.write(chunk)

    print("✅ Чанки успешно сохранены!")

if __name__ == "__main__":
    process_files()
