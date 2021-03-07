set -e

for i in {1..10}
do
    echo
    echo "RUN ${i}"
    echo
    CUDA_VISIBLE_DEVICES="3" python train.py \
    --expt=node_emb_24_l2_1.0/run_${i} --dataset='m5' --random_seed=${i} \
    --node_emb_dim=24 --fixed_lstm_hidden=24 \
    --batch_size=500 --train_epochs=25 --patience=5 --learning_rate=0.01 \
    --hist_len=28 --train_pred=28 --test_pred=7 --val_windows=5 --test_windows=5 \
    --reg_type=l2 --reg_weight=1.0
    # break  # Comment when running multiple expts
done
