# 🧠 Dokumentasi Model Machine Learning

> Sistem prediksi tip taksi NYC menggunakan **Random Forest** dalam pipeline **Kappa Architecture**.

---

## 1. Gambaran Umum

Project ini melatih dua model Machine Learning secara bersamaan dari data yang sama:

| Model | Tipe | Output | Disimpan di |
|-------|------|--------|-------------|
| **Random Forest Regressor** | Regresi | Nilai tip nominal (`$`) | `taxi_reg_model/` |
| **Random Forest Classifier** | Klasifikasi | Kategori tip (Rendah/Menengah/Tinggi) | `taxi_class_model/` |

Kedua model dilatih menggunakan **Apache Spark MLlib** dan disimpan dalam format Spark ML yang dapat dimuat kembali untuk prediksi.

---

## 2. Data Training

### Sumber Data
- **Dataset**: NYC Green Taxi Trip Records (NYC TLC)
- **Periode**: Februari 2015 – Desember 2018
- **Total baris asli**: ~54 juta perjalanan
- **Baris setelah filtering**: ~23.7 juta perjalanan
- **Filter yang diterapkan**:
  - `payment_type = '1'` → hanya kartu kredit (tip tercatat elektronik)
  - `fareAmount > 0` dan `fareAmount < 500`
  - `tripDistance > 0` dan `tripDistance < 100`
  - `passenger_count > 0`
  - `tip_amount >= 0`

### Split Data
```
Total: 23,683,316 baris
├── Training Set (80%) : 18,949,534 baris  ← digunakan untuk melatih model
└── Test Set     (20%) :  4,733,782 baris  ← digunakan untuk evaluasi
     seed = 42 (reproducible)
```

---

## 3. Label (Target Prediksi)

### ⚠️ Keputusan Penting: Mengapa `fareAmount` TIDAK digunakan sebagai fitur?

Label yang diprediksi adalah **`tip_amount`** — nilai tip asli yang tercatat dari NYC TLC.

Sebelumnya, proyek ini menggunakan rumus buatan `tip = fareAmount × 0.15` sebagai label, lalu memasukkan `fareAmount` sebagai fitur. Ini menyebabkan **Data Leakage** karena:

```
Label = fareAmount × 0.15
Fitur mengandung fareAmount
→ Model hanya belajar rumus matematika sederhana
→ Akurasi 100% (palsu!)
```

Setelah diperbaiki:
```
Label = tip_amount (nilai nyata dari NYC TLC)
fareAmount DIHAPUS dari fitur
→ Model belajar pola perilaku penumpang yang sesungguhnya
→ Akurasi 72.94% (realistis)
```

### Label Klasifikasi (Bucketizer)

Nilai `tip_amount` dikelompokkan menjadi 3 kategori:

| Kelas | Label | Rentang Tip |
|-------|-------|-------------|
| 0 | **Rendah** | $0.00 – $2.00 |
| 1 | **Menengah** | $2.01 – $5.00 |
| 2 | **Tinggi** | > $5.00 |

---

## 4. Fitur (Input Model)

| # | Fitur | Tipe | Deskripsi |
|---|-------|------|-----------|
| 1 | `tripDistance` | Float | Jarak perjalanan dalam miles |
| 2 | `passengerCount` | Integer | Jumlah penumpang dalam satu trip |
| 3 | `pickup_zone` | Float | ID zona pickup (1–263, area NYC) |
| 4 | `dropoff_zone` | Float | ID zona dropoff (1–263, area NYC) |
| 5 | `pickup_hour` | Integer | Jam pickup (0–23) |
| 6 | `pickup_day` | Integer | Hari dalam seminggu (1=Senin, 7=Minggu) |

**Fitur yang sengaja DIBUANG:**

| Fitur | Alasan |
|-------|--------|
| `fareAmount` | Berkorelasi langsung dengan `tip_amount` → data leakage |

---

## 5. Algoritma: Random Forest

### Mengapa Random Forest?

Random Forest dipilih karena:
- ✅ Tahan terhadap overfitting (ensemble dari banyak pohon)
- ✅ Tidak memerlukan normalisasi fitur
- ✅ Mendukung fitur campuran (numerik + kategorik)
- ✅ Menghasilkan **Feature Importance** untuk interpretasi
- ✅ Tersedia native di Spark MLlib (skalabel untuk jutaan baris)

### Hyperparameter

| Parameter | Nilai | Tujuan |
|-----------|-------|--------|
| `numTrees` | 50 | Jumlah pohon keputusan dalam ensemble |
| `maxDepth` | 10 | Kedalaman maksimum tiap pohon (batasi overfitting) |
| `minInstancesPerNode` | 10 | Minimum data per node (regularisasi) |
| `featureSubsetStrategy` | `sqrt` | Jumlah fitur per split = √6 ≈ 2 (regularisasi) |
| `seed` | 42 | Reproducibility |

---

## 6. Hasil Evaluasi

Model dievaluasi pada **Test Set** yang **tidak pernah dilihat** selama training.

### Model Regresi (Prediksi Nilai Tip $)

| Metrik | Nilai | Interpretasi |
|--------|-------|-------------|
| **RMSE** | $2.28 | Rata-rata error prediksi tip |
| **MAE** | $1.10 | Error absolut rata-rata |
| **R²** | 0.1978 | Model menjelaskan ~20% variasi data |

> **Catatan R²**: Nilai R² yang rendah (~0.2) wajar untuk data taksi nyata karena tip dipengaruhi faktor psikologis penumpang yang tidak bisa diukur (mood, kepuasan layanan, dll).

### Model Klasifikasi (Prediksi Kategori Tip)

| Metrik | Nilai | Interpretasi |
|--------|-------|-------------|
| **Akurasi** | 72.94% | 73 dari 100 prediksi benar |
| **F1-Score** | 0.7393 | Rata-rata harmonis precision & recall |
| **Cohen's Kappa (κ)** | **0.5756** | Kualitas model (Moderate/Cukup) |

### Confusion Matrix

```
                 Prediksi
Aktual     Rendah  Menengah  Tinggi
─────────────────────────────────────
Rendah      9,568     3,071     570
Menengah    2,191    10,059     741
Tinggi         63     1,581   2,156
```

**Analisis:**
- Model paling akurat mengenali kategori **Rendah** dan **Tinggi**
- Kategori **Menengah** ($2–$5) paling sulit diprediksi karena batas antar kelas yang tipis

---

## 7. Cohen's Kappa (κ) — Metrik Utama

**Cohen's Kappa** adalah metrik evaluasi klasifikasi yang lebih andal dari akurasi biasa karena memperhitungkan kemungkinan prediksi benar karena kebetulan.

### Formula

```
κ = (Po - Pe) / (1 - Pe)

Po = Observed Agreement (Akurasi biasa)
Pe = Expected Agreement (Peluang kebetulan)
```

### Skala Interpretasi

| Nilai κ | Kategori | Status Model Ini |
|---------|----------|-----------------|
| > 0.80 | Sangat Baik (Almost Perfect) | |
| > 0.60 | Baik (Substantial) | |
| **> 0.40** | **Cukup (Moderate)** | **← Model ini: κ = 0.5756** |
| > 0.20 | Lemah (Fair) | |
| ≤ 0.20 | Sangat Lemah | |

> κ = 0.5756 artinya model **57.6% lebih baik** dari tebakan acak.

---

## 8. Feature Importance

| Fitur | Importansi | Visual |
|-------|-----------|--------|
| `tripDistance` | **0.8777** | ████████████████████████████████████ 87.8% |
| `dropoff_zone` | 0.0779 | ███ 7.8% |
| `pickup_zone` | 0.0280 | █ 2.8% |
| `pickup_hour` | 0.0135 | 1.4% |
| `pickup_day` | 0.0021 | 0.2% |
| `passengerCount` | 0.0008 | 0.1% |

**Interpretasi:**
- `tripDistance` mendominasi dengan **87.8%** — penumpang cenderung memberi tip proporsional dengan jarak
- `dropoff_zone` dan `pickup_zone` berkontribusi karena area di NYC memiliki kebiasaan tip berbeda
- `pickup_hour` dan `pickup_day` berperan kecil — rush hour dan akhir pekan sedikit berpengaruh

---

## 9. Cara Menggunakan Model

### Via Script Siap Pakai
```bash
python predict.py         # Test prediksi beberapa contoh trip
python evaluate_model.py  # Evaluasi lengkap dengan Cohen's Kappa
bash run_all.sh           # Jalankan semua sekaligus
```

### Via Dashboard (Streamlit)
```bash
streamlit run app.py
# Akses: http://localhost:8501
```

### Load Manual di Python
```python
import os
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@17"

from pyspark.sql import SparkSession
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType

spark = SparkSession.builder.appName("TaxiPredict").getOrCreate()

model_reg   = RandomForestRegressionModel.load("./taxi_reg_model")
model_class = RandomForestClassificationModel.load("./taxi_class_model")

schema = StructType([
    StructField("tripDistance",   DoubleType()),
    StructField("passengerCount", IntegerType()),
    StructField("pickup_zone",    DoubleType()),
    StructField("dropoff_zone",   DoubleType()),
    StructField("pickup_hour",    IntegerType()),
    StructField("pickup_day",     IntegerType()),
])

# Contoh: 5 mil, 1 penumpang, zona 140→236, jam 9 pagi, Selasa
data = [(5.0, 1, 140.0, 236.0, 9, 2)]
df   = spark.createDataFrame(data, schema)

assembler = VectorAssembler(
    inputCols=["tripDistance","passengerCount","pickup_zone",
               "dropoff_zone","pickup_hour","pickup_day"],
    outputCol="features"
)
df_features = assembler.transform(df)

tip_nominal  = model_reg.transform(df_features).select("prediction").collect()[0][0]
tip_kategori = model_class.transform(df_features).select("prediction").collect()[0][0]

label_map = {0.0: "Rendah ($0-$2)", 1.0: "Menengah ($2-$5)", 2.0: "Tinggi (>$5)"}
print(f"Prediksi tip: ${tip_nominal:.2f} → {label_map[tip_kategori]}")
```

---

## 10. Keterbatasan Model

| Keterbatasan | Penjelasan |
|-------------|-----------|
| **R² rendah (0.20)** | Tip manusia dipengaruhi faktor non-teknis (mood, kualitas layanan) yang tidak ada dalam data |
| **Hanya kartu kredit** | Model dilatih dari `payment_type = '1'`. Tip tunai tidak tercatat sehingga tidak dimasukkan |
| **Data 2015–2018** | Perilaku tip bisa berubah setelah 2018. Model perlu dilatih ulang dengan data terbaru |
| **Fitur terbatas** | Tidak ada data cuaca, event kota, atau rating sopir yang bisa meningkatkan akurasi |

---

## 11. Struktur Folder Model

```
taxi_reg_model/               ← Random Forest Regressor
├── data/                     ← Serialized tree data (Parquet)
├── metadata/                 ← Model metadata (JSON)
└── treesMetadata/            ← Metadata tiap pohon

taxi_class_model/             ← Random Forest Classifier
├── data/
├── metadata/
└── treesMetadata/
```

> Model disimpan dalam format **Spark ML**. Tidak kompatibel dengan scikit-learn atau library non-Spark.

---

## 12. Riwayat Versi Model

| Versi | Tanggal | Keterangan |
|-------|---------|-----------|
| v1 (dihapus) | — | Label buatan `fareAmount × 0.15`, data leakage, akurasi 100% (tidak valid) |
| **v2 (saat ini)** | Jun 2026 | Label `tip_amount` asli, tanpa `fareAmount` di fitur, akurasi 72.94%, κ=0.5756 |
