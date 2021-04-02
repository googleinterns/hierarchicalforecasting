set -e

for i in {1..10}
do
    echo
    echo "RUN ${i}"
    echo
    CUDA_VISIBLE_DEVICES="3" python train.py \
    --expt=hid_60_emb_10_lr_0.001/run_${i} --dataset='m5' --random_seed=${i} \
    --node_emb_dim=10 --fixed_lstm_hidden=60 \
    --batch_size=500 --train_epochs=25 --patience=5 --learning_rate=0.001 \
    --hist_len=28 --train_pred=28 --test_pred=7 --val_windows=5 --test_windows=5
    # break  # Comment when running multiple expts
done
