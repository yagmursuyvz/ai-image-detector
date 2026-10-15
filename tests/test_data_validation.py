import pytest
import pandas as pd
import great_expectations as ge
from validate_manifest import validate_data

def test_corrupted_data_validation_fails(tmp_path):
    # 1. Gerçek manifest dosyasını oku
    df = pd.read_csv("data/processed/manifest.csv")
    
    # 2. Bilerek veriyi boz: İlk satırın etiketini (label) 2 yapıyoruz!
    df.loc[0, "label"] = 2
    
    # Bozulmuş veriyi geçici bir CSV dosyasına kaydet
    corrupted_csv_path = tmp_path / "corrupted_manifest.csv"
    df.to_csv(corrupted_csv_path, index=False)
    
    # 3. Great Expectations ile bozuk veriyi doğrula
    ge_df = ge.read_csv(corrupted_csv_path)
    res = ge_df.expect_column_values_to_be_in_set("label", [0, 1])
    
    # 4. Doğrulamanın BAŞARISIZ olduğunu (patladığını) kontrol et
    assert not res["success"], "Test Başarılı: Bozuk veri (label=2) beklendiği gibi Great Expectations doğrulamasını patlattı!"
    print("\n[TEST PASSED] Hatalı label=2 satırı Great Expectations tarafından başarıyla yakalandı ve doğrulama patladı!")