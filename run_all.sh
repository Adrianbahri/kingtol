#!/bin/zsh
# ============================================================
# run_all.sh — Otomatisasi Full Pipeline Kappa Architecture
# Jalankan dari root project: bash run_all.sh
# ============================================================

set -e

VENV="./venv/bin/activate"
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║     KAPPA ARCHITECTURE — Auto Pipeline Runner       ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

if [ ! -f "$VENV" ]; then
    echo "❌ venv tidak ditemukan. Jalankan: python -m venv venv && pip install -r requirements.txt"
    exit 1
fi
source "$VENV"

# ── Step 1: Training ───────────────────────────────────────
echo "┌─────────────────────────────────────────────────────┐"
echo "│  Step 1: Training Model (Kappa Retraining)          │"
echo "└─────────────────────────────────────────────────────┘"

if [ -d "./models/taxi_reg_model" ] && [ -d "./models/taxi_class_model" ]; then
    echo "✅ Model sudah ada di models/. Skip training."
    echo "   (Hapus folder models/taxi_reg_model & models/taxi_class_model untuk retrain)"
else
    echo "⏳ Melatih model... (15-30 menit untuk data besar)"
    cd ml && python spark_batch_training.py 2>&1 | tee "../$LOG_DIR/training.log"
    cd ..
    echo "✅ Training selesai! Log: $LOG_DIR/training.log"
fi

echo ""

# ── Step 2: Evaluasi ──────────────────────────────────────
echo "┌─────────────────────────────────────────────────────┐"
echo "│  Step 2: Evaluasi Model + Cohen's Kappa             │"
echo "└─────────────────────────────────────────────────────┘"
echo "⏳ Mengevaluasi model..."
cd ml && python evaluate_model.py 2>&1 | tee "../$LOG_DIR/evaluation.log"
cd ..
echo "✅ Evaluasi selesai! Log: $LOG_DIR/evaluation.log"

echo ""

# ── Step 3: Test Prediksi ─────────────────────────────────
echo "┌─────────────────────────────────────────────────────┐"
echo "│  Step 3: Test Prediksi (beberapa contoh trip)       │"
echo "└─────────────────────────────────────────────────────┘"
echo "⏳ Menjalankan prediksi contoh..."
cd ml && python predict.py 2>&1 | tee "../$LOG_DIR/predict.log"
cd ..
echo "✅ Prediksi selesai! Log: $LOG_DIR/predict.log"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  ✅ SEMUA SELESAI!                                   ║"
echo "║                                                      ║"
echo "║  Langkah selanjutnya (opsional):                     ║"
echo "║  • Dashboard  : cd dashboard && streamlit run app.py ║"
echo "║  • Streaming  : cd stream && python data_generator.py║"
echo "║                 cd stream && python kappa_stream_ingest.py ║"
echo "║                 cd stream && python realtime_predictor.py  ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
