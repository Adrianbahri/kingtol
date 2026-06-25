import clickhouse_connect

# 1. Koneksi ke ClickHouse lokal
client = clickhouse_connect.get_client(
    host='localhost', 
    port=8123, 
    username='mahasiswa', 
    password='bigdata123'
)

# 2. Buat Database
client.command('CREATE DATABASE IF NOT EXISTS taxi_db')

# 3. Buat Tabel MergeTree yang Dioptimalkan
# tip_amount ditambahkan sebagai kolom ASLI dari dataset NYC TLC
# payment_type '1' = cash (memiliki tip asli), '2' = credit card, dsb
schema_query = """
CREATE TABLE IF NOT EXISTS taxi_db.green_taxi (
    VendorID         LowCardinality(String),
    lpepPickupDatetime  DateTime,
    lpepDropoffDatetime DateTime,
    passenger_count  Float32,
    tripDistance     Float32,
    puLocationId     LowCardinality(String),
    doLocationId     LowCardinality(String),
    RatecodeID       LowCardinality(String),
    payment_type     LowCardinality(String),
    fareAmount       Float32,
    tip_amount       Float32
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(lpepPickupDatetime)
ORDER BY (puLocationId, lpepPickupDatetime)
SETTINGS index_granularity = 8192;
"""
client.command(schema_query)
print("Tabel green_taxi berhasil dibuat dengan kolom tip_amount!")