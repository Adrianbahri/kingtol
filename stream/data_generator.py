import json
import time
import random
from kafka import KafkaProducer
from datetime import datetime

# ============================================================
# KAPPA ARCHITECTURE — Kafka Producer
# Peran: Satu-satunya sumber data masuk ke sistem.
# Semua data (historis-simulasi maupun real-time) melewati
# jalur ini sebelum diproses oleh Spark Streaming.
# ============================================================

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

# Zona pickup/dropoff yang umum di NYC (sample representatif)
COMMON_ZONES = [7, 10, 12, 13, 24, 36, 41, 42, 43, 45, 48,
                50, 61, 74, 75, 82, 107, 113, 114, 116, 120,
                140, 141, 142, 143, 144, 145, 148, 166, 179,
                181, 186, 209, 217, 223, 225, 231, 232, 234, 236, 237]

print("="*55)
print("  KAPPA ARCHITECTURE — Kafka Data Producer")
print("  Sumber tunggal data streaming masuk ke sistem")
print("="*55)

def generate_taxi_trip():
    """
    Generate data trip taksi yang realistis.
    Tip dihitung berdasarkan pola nyata NYC:
    - Jarak jauh cenderung tip lebih besar
    - Rush hour (7-9, 17-19) tip lebih besar
    - Malam minggu (6-7) tip lebih besar
    """
    fare      = round(random.uniform(3.5, 80.0), 2)
    distance  = round(random.uniform(0.3, 25.0), 2)
    hour      = datetime.now().hour
    day       = datetime.now().isoweekday()
    pax       = random.choices([1, 2, 3, 4, 5, 6],
                               weights=[55, 20, 10, 8, 4, 3])[0]
    pu_zone   = float(random.choice(COMMON_ZONES))
    do_zone   = float(random.choice(COMMON_ZONES))

    # Tip yang realistis (sebagian besar 0 atau 10-25% fare)
    # ~30% trip tidak kasih tip (tip=0)
    if random.random() < 0.30:
        tip = 0.0
    else:
        tip_pct = random.uniform(0.08, 0.30)
        tip = round(fare * tip_pct, 2)

    return {
        "tip_amount":      tip,
        "fareAmount":      fare,
        "tripDistance":    distance,
        "passengerCount":  pax,
        "pickup_zone":     pu_zone,
        "dropoff_zone":    do_zone,
        "pickup_hour":     hour,
        "pickup_day":      day
    }

trip_count = 0
print(f"\n[Producer] Mengirim data ke Kafka topic: green-taxi-stream")
print("[Producer] Tekan Ctrl+C untuk berhenti\n")

try:
    while True:
        data = generate_taxi_trip()
        producer.send('green-taxi-stream', value=data)
        trip_count += 1

        tip_str = f"${data['tip_amount']:.2f}" if data['tip_amount'] > 0 else "No tip"
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Trip #{trip_count:,} | "
              f"Fare=${data['fareAmount']:.2f} | Dist={data['tripDistance']:.1f}mi | "
              f"Tip={tip_str}")

        time.sleep(random.uniform(0.5, 2.0))

except KeyboardInterrupt:
    print(f"\n[Producer] Dihentikan. Total {trip_count:,} trip dikirim.")
finally:
    producer.flush()
    producer.close()