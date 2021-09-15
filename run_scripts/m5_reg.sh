set -e

for i in {1..10}
do
    echo
    echo "RUN ${i}"
    echo
    CUDA_VISIBLE_DEVICES="1" python train.py \
    --expt=hid_64_emb_10_timear/run_${i} --dataset='m5' --random_seed=${i} \
    --fixed_lstm_hidden=64 --node_emb_dim=10 --emb_reg_weight=1e-6 \
    --batch_size=500 --train_epochs=40 --patience=10 --learning_rate=0.01 --num_changes=6 \
    --hist_len=28 --train_pred=7 --test_pred=7 --val_windows=5 --test_windows=5
done
