set -e

for i in {1..10}
do
    echo
    echo "RUN ${i}"
    echo
    CUDA_VISIBLE_DEVICES="2" python train.py \
    --expt=hid_90_exp_1/run_${i} --dataset='favorita' --random_seed=${i} \
    --fixed_lstm_hidden=90 --node_emb_dim=1 \
    --batch_size=500 --train_epochs=30 --patience=10 --learning_rate=0.001 --num_changes=6 \
    --hist_len=28 --train_pred=28 --test_pred=7 --val_windows=5 --test_windows=5
done
