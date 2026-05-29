#!/bin/bash
# Comprehensive TimeMixer Test Suite
# Runs all ETT long-term forecasting benchmarks and logs results

RESULTS_DIR="./test_results_nightly"
mkdir -p "$RESULTS_DIR"
SUMMARY="$RESULTS_DIR/summary.txt"
echo "TimeMixer Comprehensive Test Report" > "$SUMMARY"
echo "Started: $(date)" >> "$SUMMARY"
echo "========================================" >> "$SUMMARY"

run_test() {
    local dataset=$1
    local data_class=$2
    local enc_in=$3
    local pred_len=$4
    local root_path=$5
    local data_path=$6
    local extra_args=$7

    local model_id="${dataset}_96_${pred_len}"
    local logfile="$RESULTS_DIR/${model_id}.log"

    echo ">>> Testing $model_id at $(date)" | tee -a "$SUMMARY"

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path "$root_path" \
      --data_path "$data_path" \
      --model_id "$model_id" \
      --model TimeMixer \
      --data "$data_class" \
      --features M \
      --seq_len 96 \
      --label_len 0 \
      --pred_len "$pred_len" \
      --e_layers 2 \
      --enc_in "$enc_in" \
      --c_out "$enc_in" \
      --des 'Exp' \
      --itr 1 \
      --d_model 16 \
      --d_ff 32 \
      --learning_rate 0.01 \
      --train_epochs 10 \
      --patience 10 \
      --batch_size 128 \
      --down_sampling_layers 3 \
      --down_sampling_method avg \
      --down_sampling_window 2 \
      $extra_args 2>&1 | tee "$logfile"

    # Extract MSE and MAE
    local mse=$(grep "mse:" "$logfile" | tail -1 | grep -oP 'mse:\K[0-9.]+')
    local mae=$(grep "mae:" "$logfile" | tail -1 | grep -oP 'mae:\K[0-9.]+')
    echo "  Result: MSE=$mse MAE=$mae" | tee -a "$SUMMARY"
    echo "" >> "$SUMMARY"
}

echo "" >> "$SUMMARY"
echo "========================================" >> "$SUMMARY"
echo "LONG-TERM FORECASTING - ETT DATASETS" >> "$SUMMARY"
echo "========================================" >> "$SUMMARY"

# ETTh1 (7 features)
for pred_len in 96 192 336 720; do
    run_test "ETTh1" "ETTh1" 7 "$pred_len" "./dataset/ETT-small/" "ETTh1.csv"
done

# ETTh2 (7 features)
for pred_len in 96 192 336 720; do
    run_test "ETTh2" "ETTh2" 7 "$pred_len" "./dataset/ETT-small/" "ETTh2.csv"
done

# ETTm1 (7 features)
for pred_len in 96 192 336 720; do
    run_test "ETTm1" "ETTm1" 7 "$pred_len" "./dataset/ETT-small/" "ETTm1.csv"
done

# ETTm2 (7 features)
for pred_len in 96 192 336 720; do
    run_test "ETTm2" "ETTm2" 7 "$pred_len" "./dataset/ETT-small/" "ETTm2.csv"
done

echo "" >> "$SUMMARY"
echo "========================================" >> "$SUMMARY"
echo "Completed: $(date)" >> "$SUMMARY"
echo "Full summary saved to $SUMMARY"
cat "$SUMMARY"
