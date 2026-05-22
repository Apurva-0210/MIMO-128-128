#!/usr/bin/env bash
# =====================================================================
#  train_all.sh — Train every algorithm for both array sizes
# =====================================================================
#  Trains in this order (stages and array sizes):
#
#    STAGE I (Jakes screening):
#      1. CNN-LSTM     — 128x128, 64x64
#      2. CNN-Sparse   — 128x128, 64x64
#      3. Sa-DLCS      — 128x128, 64x64  (Jakes)
#      4. NCRN         — 128x128, 64x64  (Jakes)
#
#    STAGE II (TDL validation, all four retrained):
#      5. CNN-LSTM     — 128x128, 64x64  (TDL)
#      6. CNN-Sparse   — 128x128, 64x64  (TDL)
#      7. Sa-DLCS      — 128x128, 64x64  (TDL)
#      8. NCRN         — 128x128, 64x64  (TDL)
#
#  Total: 16 trained models. Approx runtime on Apple M1:
#    - 64x64 models:   ~3-5 min each   (~40 min total)
#    - 128x128 models: ~10-15 min each (~120 min total)
#  Full pipeline: ~2.5 hours.
#
#  Skip parts by passing array argument:
#    bash train_all.sh 64    # only 64x64
#    bash train_all.sh 128   # only 128x128
#  Default: both sizes.
# =====================================================================

set -e   # fail on any error
cd "$(dirname "$0")"

ARRAY_SIZES="${1:-both}"
if [ "$ARRAY_SIZES" = "both" ]; then
    SIZES="64 128"
elif [ "$ARRAY_SIZES" = "64" ]; then
    SIZES="64"
elif [ "$ARRAY_SIZES" = "128" ]; then
    SIZES="128"
else
    echo "Usage: bash train_all.sh [64|128|both]"
    exit 1
fi

echo "======================================================================"
echo "  TRAINING PIPELINE — array sizes: $SIZES"
echo "======================================================================"

START_TIME=$(date +%s)

for SZ in $SIZES; do
    echo ""
    echo "######################################################################"
    echo "###                                                                ###"
    echo "###  ARRAY SIZE: ${SZ}x${SZ}                                              ###"
    echo "###                                                                ###"
    echo "######################################################################"

    # Stage I: Jakes
    echo ""
    echo "─── [1/8] CNN-LSTM Jakes ${SZ}x${SZ} ───"
    cd stage1_jakes/cnn_lstm
    python train_cnn_lstm.py --array $SZ
    cd ../..

    echo ""
    echo "─── [2/8] CNN-Sparse Jakes ${SZ}x${SZ} ───"
    cd stage1_jakes/cnn_sparse
    python train_cnn_sparse.py --array $SZ
    cd ../..

    echo ""
    echo "─── [3/8] Sa-DLCS Jakes ${SZ}x${SZ} ───"
    cd stage1_jakes/sa_dlcs
    python train_sa_dlcs_jakes.py --array $SZ
    cd ../..

    echo ""
    echo "─── [4/8] NCRN Jakes ${SZ}x${SZ} ───"
    cd stage1_jakes/ncrn
    python train_ncrn_jakes.py --array $SZ
    cd ../..

    # Stage II: TDL — all four retrained
    echo ""
    echo "─── [5/8] CNN-LSTM TDL ${SZ}x${SZ} ───"
    cd stage2_tdl/cnn_lstm
    python train_cnn_lstm_tdl.py --array $SZ
    cd ../..

    echo ""
    echo "─── [6/8] CNN-Sparse TDL ${SZ}x${SZ} ───"
    cd stage2_tdl/cnn_sparse
    python train_cnn_sparse_tdl.py --array $SZ
    cd ../..

    echo ""
    echo "─── [7/8] Sa-DLCS TDL ${SZ}x${SZ} ───"
    cd stage2_tdl/sa_dlcs
    python train_sa_dlcs_tdl.py --array $SZ
    cd ../..

    echo ""
    echo "─── [8/8] NCRN TDL ${SZ}x${SZ} ───"
    cd stage2_tdl/ncrn
    python train_ncrn_tdl.py --array $SZ
    cd ../..
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MIN=$((ELAPSED / 60))

echo ""
echo "======================================================================"
echo "  TRAINING COMPLETE — total time: ${MIN} min"
echo "  Models saved in: ./models/"
echo ""
echo "  Next step: generate figures"
echo "    python plotting/plot_all_figures.py"
echo "    python plotting/plot_all_figures.py --quick   # for fast preview"
echo "======================================================================"
