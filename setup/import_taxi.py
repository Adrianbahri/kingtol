import pandas as pd
import clickhouse_connect
from datetime import datetime
from dateutil.relativedelta import relativedelta
import gc
import time
import urllib.request
import tempfile
import os


print("Connect ClickHouse...")

client = clickhouse_connect.get_client(
    host="localhost",
    port=8123,
    username="mahasiswa",
    password="bigdata123",
    database="taxi_db"
)


# Range data yang akan diimport
mulai = datetime(2015, 2, 1)
selesai = datetime(2018, 12, 31)

tanggal = mulai

TIMEOUT = 120  # timeout 120 detik per download


def download_parquet(url, timeout=120):
    """Download parquet file dengan timeout, return DataFrame."""
    tmp_path = None
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".parquet")
            with os.fdopen(tmp_fd, 'wb') as f:
                while True:
                    chunk = resp.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    f.write(chunk)
        df = pd.read_parquet(tmp_path)
        return df
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


while tanggal <= selesai:

    akhir = tanggal + relativedelta(months=3) - relativedelta(days=1)

    if akhir > selesai:
        akhir = selesai

    print(f"\nAmbil data {tanggal.date()} - {akhir.date()}")

    try:
        frames = []
        bulan = tanggal

        while bulan <= akhir:
            y = bulan.year
            m = bulan.month

            url = (
                f"https://d37ci6vzurychx.cloudfront.net/"
                f"trip-data/green_tripdata_{y}-{m:02d}.parquet"
            )

            print(f"  Download {y}-{m:02d}...")

            try:
                df_bulan = download_parquet(url, timeout=TIMEOUT)
                frames.append(df_bulan)
                print(f"  OK: {len(df_bulan):,} baris")
            except Exception as e_dl:
                print(f"  Skip {y}-{m:02d}: {e_dl}")

            bulan += relativedelta(months=1)

        if len(frames) == 0:
            print("  Tidak ada data periode ini")
            tanggal += relativedelta(months=3)
            continue

        df = pd.concat(frames, ignore_index=True)
        del frames

        print(f"  Total: {len(df):,} baris")

        # sesuaikan nama kolom (NYC TLC format -> ClickHouse schema)
        df = df.rename(columns={
            "lpep_pickup_datetime":  "lpepPickupDatetime",
            "lpep_dropoff_datetime": "lpepDropoffDatetime",
            "passenger_count":       "passenger_count",
            "trip_distance":         "tripDistance",
            "PULocationID":          "puLocationId",
            "DOLocationID":          "doLocationId",
            "RatecodeID":            "RatecodeID",
            "payment_type":          "payment_type",
            "fare_amount":           "fareAmount",
            "tip_amount":            "tip_amount",   # ← Kolom tip ASLI
        })

        kolom = [
            "VendorID",
            "lpepPickupDatetime",
            "lpepDropoffDatetime",
            "passenger_count",
            "tripDistance",
            "puLocationId",
            "doLocationId",
            "RatecodeID",
            "payment_type",
            "fareAmount",
            "tip_amount",       # ← Sertakan tip_amount asli
        ]

        # hanya ambil kolom yang ada
        kolom_ada = [c for c in kolom if c in df.columns]
        df = df[kolom_ada].copy()

        # Pastikan tip_amount ada (default 0 jika tidak ada di dataset lama)
        if "tip_amount" not in df.columns:
            df["tip_amount"] = 0.0

        # cleaning numerik
        for c in ["passenger_count", "tripDistance", "fareAmount", "tip_amount"]:
            if c in df.columns:
                df[c] = df[c].fillna(0).astype(float)

        for c in ["VendorID", "puLocationId", "doLocationId",
                  "RatecodeID", "payment_type"]:
            if c in df.columns:
                df[c] = df[c].fillna("Unknown").astype(str)

        if "lpepPickupDatetime" in df.columns:
            df["lpepPickupDatetime"] = pd.to_datetime(df["lpepPickupDatetime"])

        if "lpepDropoffDatetime" in df.columns:
            df["lpepDropoffDatetime"] = pd.to_datetime(df["lpepDropoffDatetime"])

        print("  Insert ke ClickHouse...")
        client.insert_df("green_taxi", df)
        print(f"  Berhasil: {len(df):,} baris masuk")

    except Exception as e:
        print(f"ERROR: {e}")
        time.sleep(5)
        continue

    del df
    gc.collect()

    tanggal += relativedelta(months=3)


print("\nSELESAI - Semua data masuk ClickHouse (termasuk tip_amount asli)")