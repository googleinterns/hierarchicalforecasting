set -e

for i in {1..10}
do
    echo
    echo "RUN ${i}"
    echo
    CUDA_VISIBLE_DEVICES="3" python train.py \
    --expt=hid_96_exp_1_norm/run_${i} --dataset='wiki' --random_seed=${i} \
    --fixed_lstm_hidden=96 --node_emb_dim=1 \
    --batch_size=512 --train_epochs=40 --patience=10 --learning_rate=0.01 --num_changes=6 \
    --hist_len=28 --train_pred=28 --test_pred=7 --val_windows=5 --test_windows=5
done
