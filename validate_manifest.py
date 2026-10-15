import great_expectations as ge


def validate_data():
    print("Manifest dosyası Great Expectations ile doğrulanıyor...")
    
    # Manifest dosyasını okuyoruz (GE DataFrame kullanarak)
    df = ge.read_csv("data/processed/manifest.csv")
    
    # 1. Gerekli sütunların varlığını kontrol et
    expected_columns = ["file_path", "label", "width", "height", "format", "split", "file_size_bytes", "hash"]
    for col in expected_columns:
        res = df.expect_column_to_exist(col)
        assert res["success"], f"Eksik sütun: {col}"
        
    # 2. Kritik alanlarda null (boş) değer var mı?
    for col in expected_columns:
        res = df.expect_column_values_to_not_be_null(col)
        assert res["success"], f"Kritik alanda null değer bulundu: {col}"
        
    # 3. Etiketler sadece {0, 1} aralığında mı?
    res = df.expect_column_values_to_be_in_set("label", [0, 1])
    assert res["success"], "Etiketler yalnızca 0 veya 1 olmalıdır!"
    
    # 4. Tekrar eden hash (kopya görsel) var mı kontrolü (uyarı amaçlı)
    res = df.expect_column_values_to_be_unique("hash")
    if not res["success"]:
        print(f"Uyarı: Tekrar eden (kopya) görseller tespit edildi! Toplam benzersiz olmayan: {res['result']['unexpected_count']}")
        
    print("Tüm Great Expectations doğrulamaları başarıyla geçti! Veri seti temiz.")

if __name__ == "__main__":
    validate_data()