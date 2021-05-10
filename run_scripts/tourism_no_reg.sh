set -e

for i in {1..10}
do
    echo
    echo "RUN ${i}"
    echo
    CUDA_VISIBLE_DEVICES="2" python train.py \
    --expt=hid_10_exp_4_norm/run_${i} --dataset='tourism' --random_seed=${i} \
    --fixed_lstm_hidden=10 --node_emb_dim=4 \
    --batch_size=512 --train_epochs=40 --patience=10 --learning_rate=0.1 --num_changes=6 \
    --hist_len=24 --train_pred=4 --test_pred=4 --val_windows=3 --test_windows=3
done
