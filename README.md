# 🚖 NYC Green Taxi Tip Prediction Platform

Platform analitik dan prediksi tip taksi NYC skala besar menggunakan **Kappa Architecture** — satu jalur streaming tunggal dari Kafka hingga prediksi real-time berbasis Machine Learning.

> **Tugas Akhir Big Data** — Universitas Hasanuddin

---

## 🏗️ Arsitektur: Kappa Architecture

```
[NYC TLC Dataset / Simulator]
         │
         ▼
  [data_generator.py]          ← Kafka Producer (satu-satunya pintu masuk data)
         │
         ▼
    [Apache Kafka]              ← Message broker (green-taxi-stream topic)
    /            \
   ▼              ▼
[kappa_stream_ingest.py]    [realtime_predictor.py]
(Stream → ClickHouse)        (Stream → ML Prediction)
   │
   ▼
[ClickHouse]                  ← Serving Layer (Single Source of Truth)
   │
   ▼
[spark_batch_training.py]     ← Periodic Retraining dari ClickHouse
   │
   ▼
[ML Model]  ←─────────────── dipakai oleh realtime_predictor.py & app.py
```

> ⚡ **Kappa vs Lambda**: Tidak ada Batch Layer terpisah. Semua data masuk hanya melalui Kafka Stream.

---

## 🧠 Machine Learning

| Aspek | Detail |
|-------|--------|
| **Algoritma** | Random Forest (Regresi + Klasifikasi) |
| **Label** | `tip_amount` asli dari NYC TLC (**bukan** rumus buatan) |
| **Fitur** | `tripDistance`, `passengerCount`, `pickup_zone`, `dropoff_zone`, `pickup_hour`, `pickup_day` |
| **Fitur Dibuang** | `fareAmount` — dihapus untuk mencegah **data leakage** |
| **Metrik Evaluasi** | RMSE, MAE, R² (regresi) · Akurasi, F1, **Cohen's Kappa (κ)** (klasifikasi) |
| **Kategori Tip** | 0=Rendah ($0–$2) · 1=Menengah ($2–$5) · 2=Tinggi (>$5) |
| **Data Training** | ~54 juta baris Green Taxi NYC 2015–2018 |

### Cohen's Kappa (κ) — Metrik Utama

Metrik yang digunakan untuk mengukur kualitas klasifikasi dengan memperhitungkan peluang kebetulan:

```
κ = (Po - Pe) / (1 - Pe)
  Po = Observed Agreement (Akurasi)
  Pe = Expected Agreement by Chance
```

| Nilai κ | Interpretasi |
|---------|-------------|
| > 0.8 | Sangat Baik (Almost Perfect) |
| > 0.6 | Baik (Substantial) |
| > 0.4 | Cukup (Moderate) |
| > 0.2 | Lemah (Fair) |
| ≤ 0.2 | Sangat Lemah |

---

## 🛠️ Stack Teknologi

| Teknologi | Peran |
|-----------|-------|
| **Apache Kafka + ZooKeeper** | Message broker — jalur masuk data tunggal (Kappa) |
| **Apache Spark (PySpark MLlib)** | Stream processing + Machine Learning |
| **ClickHouse** | Serving Layer — database kolomar berkinerja tinggi |
| **Grafana** | Dashboard visualisasi analitik |
| **Streamlit** | Dashboard prediksi interaktif |
| **Docker + Docker Compose** | Orkestrasi seluruh layanan infrastruktur |

---

## 🔧 Prasyarat

1. **Docker Desktop** — [Download](https://www.docker.com/products/docker-desktop/)
2. **Python 3.11** — [Download](https://www.python.org/downloads/)
   > ⚠️ Gunakan tepat Python 3.11. Versi 3.12+ bisa error di beberapa library ML.
3. **Java 17** (untuk PySpark lokal)
   ```bash
   # macOS
   brew install openjdk@17
   ```
4. **Git**

---

## 🚀 Instalasi & Setup

### 1. Clone Repositori
```bash
git clone <url-repositori>
cd "BIG DATA AKHIR"
```

### 2. Setup Virtual Environment
```bash
# macOS / Linux
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

```powershell
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Jalankan Docker Services
```bash
docker compose up -d
docker compose ps   # verifikasi semua container UP
```

| Container | Port | Akses | Kredensial |
|-----------|------|-------|------------|
| ClickHouse | `8123` / `9000` | HTTP / TCP | `mahasiswa` / `bigdata123` |
| Spark Master | `8080` / `7077` | Web UI / Cluster | — |
| Grafana | `3000` | Dashboard | `admin` / `admin123` |
| Kafka | `9092` | Broker | — |

### 4. Buat Tabel ClickHouse
```bash
python create_table.py
```

### 5. Import Data Historis (NYC TLC 2015–2018)
```bash
python import_taxi.py
```
> ⏳ Proses ini mengunduh ~54 juta baris dari NYC TLC cloud. Butuh waktu 30–60 menit tergantung koneksi internet.

---

## ▶️ Menjalankan Pipeline

### Cara Cepat — Otomatis (Setelah Data Ada di ClickHouse)
```bash
bash run_all.sh
```
Script ini akan otomatis:
1. ✅ Training model (skip jika sudah ada)
2. ✅ Evaluasi model + Cohen's Kappa
3. ✅ Test prediksi beberapa contoh trip

---

### Cara Manual — Per Komponen

#### A. Training Model
```bash
python spark_batch_training.py
```
> ⏳ ~15–30 menit untuk 54 juta baris. Output: `taxi_reg_model/` + `taxi_class_model/`

#### B. Evaluasi Model
```bash
python evaluate_model.py
```
Output: RMSE, MAE, R², Akurasi, F1-Score, **Cohen's Kappa (κ)**, Confusion Matrix

#### C. Pipeline Real-time (Kappa Stream)
Buka **3 terminal terpisah**:

```bash
# Terminal 1 — Kafka Producer (kirim data ke stream)
python data_generator.py

# Terminal 2 — Stream Ingest (Kafka → ClickHouse)
python kappa_stream_ingest.py

# Terminal 3 — Real-time Prediction (Kafka → ML Prediction)
python realtime_predictor.py
```

#### D. Dashboard Interaktif (Streamlit)
```bash
streamlit run app.py
# Buka: http://localhost:8501
```

#### E. Dashboard Monitoring (Grafana)
1. Buka `http://localhost:3000`
2. Login: `admin` / `admin123`
3. Tambah Data Source → ClickHouse
   - Server: `clickhouse` | Port: `9000`
   - Database: `taxi_db` | User: `mahasiswa` | Pass: `bigdata123`

---

## 🗂️ Struktur Folder & File

```
BIG DATA AKHIR/
├── docker-compose.yml
├── requirements.txt
├── run_all.sh
├── README.md
│
├── create_table.py
├── import_taxi.py
│
├── data_generator.py
├── kappa_stream_ingest.py
├── realtime_predictor.py
│
├── spark_batch_training.py
├── evaluate_model.py
├── predict.py
├── app.py
│
├── taxi_reg_model/
├── taxi_class_model/
├── clickhouse-config/
├── logs/
└── venv/
```

### Penjelasan Detail Setiap File

---

#### ⚙️ Konfigurasi Infrastruktur

| File | Penjelasan |
|------|-----------|
| `docker-compose.yml` | Mendefinisikan dan mengorkestrasi seluruh layanan Docker: ClickHouse, Kafka, ZooKeeper, Spark Master, Spark Worker, dan Grafana. Jalankan sekali dengan `docker compose up -d`. |
| `requirements.txt` | Daftar seluruh library Python yang dibutuhkan: PySpark, Kafka-Python, ClickHouse Connect, Pandas, Streamlit, dll. Install dengan `pip install -r requirements.txt`. |
| `run_all.sh` | Script otomasi satu klik. Menjalankan training (skip jika model sudah ada), evaluasi, dan test prediksi secara berurutan. Cukup jalankan `bash run_all.sh`. |
| `clickhouse-config/` | Folder konfigurasi tambahan untuk ClickHouse (user, password, storage limit). Digunakan otomatis oleh Docker Compose. |

---

#### 🔧 Setup & Inisialisasi Database

| File | Penjelasan |
|------|-----------|
| `create_table.py` | Membuat database `taxi_db` dan tabel `green_taxi` di ClickHouse dengan skema yang benar, termasuk kolom `tip_amount` (nilai tip asli dari NYC TLC). **Jalankan sekali di awal sebelum import data.** |
| `import_taxi.py` | Mengunduh data Green Taxi NYC 2015–2018 langsung dari server cloud NYC TLC (format `.parquet`), membersihkan data, lalu menyimpannya ke ClickHouse. Mengimpor ~54 juta baris. **Jalankan sekali setelah `create_table.py`.** |

---

#### 🌊 Kappa Stream Layer (Pipeline Utama)

Tiga file ini membentuk jalur streaming tunggal sesuai Kappa Architecture. Tidak ada batch layer terpisah.

| File | Peran dalam Kappa | Penjelasan |
|------|-------------------|-----------|
| `data_generator.py` | **Kafka Producer** | Pintu masuk satu-satunya untuk data baru ke dalam sistem. Menghasilkan data trip taksi realistis (termasuk 30% trip tanpa tip) dan mengirimkannya ke Kafka topic `green-taxi-stream` setiap 0.5–2 detik. |
| `kappa_stream_ingest.py` | **Stream → Serving Layer** | Membaca stream data dari Kafka menggunakan Spark Structured Streaming, lalu menyimpannya ke ClickHouse (Serving Layer) setiap 10 detik via `foreachBatch`. Ini satu-satunya cara data masuk ke database setelah import historis. |
| `realtime_predictor.py` | **Stream → Prediksi** | Membaca stream data dari Kafka secara paralel, lalu langsung memprediksi nilai tip menggunakan model yang sudah dilatih. Output ditampilkan ke konsol secara real-time setiap 5 detik. |

> 💡 Untuk mode real-time penuh, jalankan ketiga file ini di terminal terpisah secara bersamaan.

---

#### 🧠 Machine Learning

| File | Penjelasan |
|------|-----------|
| `spark_batch_training.py` | Membaca data dari ClickHouse (Serving Layer yang diisi oleh stream), lalu melatih dua model Random Forest: **Regresi** (prediksi nilai tip nominal dalam $) dan **Klasifikasi** (prediksi kategori Rendah/Menengah/Tinggi). Menyertakan evaluasi lengkap termasuk **Cohen's Kappa**. Output: folder `taxi_reg_model/` dan `taxi_class_model/`. |
| `evaluate_model.py` | Memuat model yang sudah tersimpan dan mengevaluasinya pada 30.000 baris data dari ClickHouse. Menampilkan RMSE, MAE, R², Akurasi, F1-Score, **Cohen's Kappa (κ)**, Confusion Matrix, dan Feature Importance. Deteksi overfitting otomatis. |
| `predict.py` | Script sederhana untuk test prediksi manual dengan beberapa contoh data trip. Berguna untuk verifikasi cepat bahwa model berjalan benar setelah training. |

---

#### 📊 Dashboard

| File | Penjelasan |
|------|-----------|
| `app.py` | Dashboard web interaktif menggunakan Streamlit. User bisa memasukkan data perjalanan (jarak, penumpang, zona, jam, hari) dan mendapatkan prediksi tip nominal ($) beserta kategorinya secara real-time. Jalankan dengan `streamlit run app.py`, akses di `http://localhost:8501`. |

---

#### 💾 Folder Output (dibuat otomatis)

| Folder | Isi |
|--------|-----|
| `taxi_reg_model/` | Model Random Forest Regressor yang sudah dilatih. Digunakan oleh `predict.py`, `realtime_predictor.py`, dan `app.py`. |
| `taxi_class_model/` | Model Random Forest Classifier yang sudah dilatih. Digunakan oleh `predict.py`, `realtime_predictor.py`, dan `app.py`. |
| `logs/` | File log hasil training (`training.log`), evaluasi (`evaluation.log`), dan prediksi (`predict.log`). Dibuat otomatis oleh `run_all.sh`. |
| `venv/` | Virtual environment Python. Tidak perlu disentuh manual. |

---

## 🔍 Verifikasi Data di ClickHouse

```bash
docker exec -it clickhouse clickhouse-client --user mahasiswa --password bigdata123
```

```sql
-- Total baris
SELECT count() FROM taxi_db.green_taxi;

-- Cek kolom tip_amount tersedia
SELECT avg(tip_amount), max(tip_amount), min(tip_amount)
FROM taxi_db.green_taxi WHERE payment_type = '1';

-- Distribusi tip per kategori
SELECT
    multiIf(tip_amount < 2, 'Rendah', tip_amount < 5, 'Menengah', 'Tinggi') AS kategori,
    count() AS jumlah,
    round(avg(tip_amount), 2) AS avg_tip
FROM taxi_db.green_taxi
WHERE payment_type = '1' AND tip_amount >= 0
GROUP BY kategori ORDER BY avg_tip;
```

---

## ⚠️ Catatan Penting

> **Mengapa `fareAmount` tidak digunakan sebagai fitur?**
>
> Tip penumpang secara statistik berkorelasi tinggi dengan `fareAmount` (biasanya 10–25% dari fare). Jika `fareAmount` dimasukkan sebagai fitur, model bisa "menghafal" korelasi tersebut dan memberikan akurasi palsu >98% — ini disebut **data leakage**. Model yang baik harus belajar dari pola perilaku penumpang, bukan dari rumus matematis.

> **Akurasi yang wajar untuk data nyata:**
> - Klasifikasi: 65–85%
> - R² Regresi: 0.3–0.7
> - Cohen's Kappa: 0.4–0.7 (Moderate–Substantial)
>
> Jika akurasi >98%, kemungkinan besar masih ada data leakage!
