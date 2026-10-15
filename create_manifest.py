import hashlib
import os
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


def get_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def generate_manifest(base_dir="data/raw/cifake", output_path="data/processed/manifest.csv"):
    data = []
    base_path = Path(base_dir)
    
    # CIFACE yapısı genellikle train/test ve içinde REAL/FAKE veya 0/1 klasörleri barındırır
    # Klasör yapına göre tarama işlemi:
    for split in ["train", "test"]:
        split_path = base_path / split
        if not split_path.exists():
            continue
            
        # Sınıf klasörlerini dönüyoruz (örneğin REAL ve FAKE veya 0 ve 1)
        for label_name in os.listdir(split_path):
            label_dir = split_path / label_name
            if not label_dir.is_dir():
                continue
                
            # Etiket ataması (Fake/Real veya 0/1 durumuna göre ayarlanabilir)
            # Genellikle CIFAKE'de 'FAKE' ve 'REAL' klasörleri olur
            label = 1 if label_name.upper() in ["FAKE", "1"] else 0
            
            for img_file in tqdm(list(label_dir.glob("*.*")), desc=f"Processing {split}/{label_name}"):
                if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue
                try:
                    with Image.open(img_file) as img:
                        width, height = img.size
                        img_format = img.format
                    
                    file_size = img_file.stat().st_size
                    file_hash = get_file_hash(img_file)
                    
                    data.append({
                        "file_path": str(img_file.relative_to(base_path.parent.parent)),
                        "label": label,
                        "width": width,
                        "height": height,
                        "format": img_format,
                        "split": split,
                        "file_size_bytes": file_size,
                        "hash": file_hash
                    })
                except Exception as e:  # noqa: BLE001
                    print(f"Hata {img_file}: {e}")
                    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Manifest dosyası başarıyla oluşturuldu: {output_path} ({len(df)} görsel kaydedildi)")

if __name__ == "__main__":
    generate_manifest()