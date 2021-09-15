set -e

for i in {1..10}
do
    echo
    echo "RUN ${i}"
    echo
    CUDA_VISIBLE_DEVICES="1" python train.py \
    --expt=hid_64_emb_10_timear/run_${i} --dataset='favorita' --random_seed=${i} \
    --fixed_lstm_hidden=20 --node_emb_dim=8 --emb_reg_weight=4.921e-4 \
    --batch_size=500 --train_epochs=40 --patience=10 --learning_rate=0.01287 --num_changes=6 \
    --hist_len=28 --train_pred=7 --test_pred=7 --val_windows=5 --test_windows=5
done
