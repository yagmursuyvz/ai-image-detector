import os
import pytest
import pandas as pd
import great_expectations as ge

def test_corrupted_data_validation_fails(tmp_path):
    manifest_path = "data/processed/manifest.csv"
    
    # Eğer CI ortamında manifest.csv dosyası henüz yoksa testi atla
    if not os.path.exists(manifest_path):
        pytest.skip("Manifest dosyası bulunamadı (CI ortamı), test atlanıyor.")
        
    # Gerçek manifest dosyasını oku
    df = pd.read_csv(manifest_path)
    
    # Bilerek veriyi boz: İlk satırın etiketini (label) 2 yapıyoruz!
    df.loc[0, "label"] = 2
    
    # Bozulmuş veriyi geçici bir CSV dosyasına kaydet
    corrupted_csv_path = tmp_path / "corrupted_manifest.csv"
    df.to_csv(corrupted_csv_path, index=False)
    
    # Great Expectations ile bozuk veriyi doğrula
    ge_df = ge.read_csv(corrupted_csv_path)
    res = ge_df.expect_column_values_to_be_in_set("label", [0, 1])
    
    # Doğrulamanın BAŞARISIZ olduğunu (patladığını) kontrol et
    assert not res["success"], "Test Başarılı: Bozuk veri Great Expectations'ı patlattı!"