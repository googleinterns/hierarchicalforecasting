set -e

for i in {1..10}
do
    echo
    echo "RUN ${i}"
    echo
    CUDA_VISIBLE_DEVICES="2" python train.py \
    --expt=hid_64_emb_8_norm/run_${i} --dataset='m5' --random_seed=${i} \
    --node_emb_dim=8 --fixed_lstm_hidden=64 \
    --batch_size=500 --train_epochs=40 --patience=10 --learning_rate=0.01 --num_changes=6 \
    --hist_len=28 --train_pred=28 --test_pred=7 --val_windows=5 --test_windows=5
    # break  # Comment when running multiple expts
done
