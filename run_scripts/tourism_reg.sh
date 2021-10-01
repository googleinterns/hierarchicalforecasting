set -e

for i in {1..10}
do
    echo
    echo "RUN ${i}"
    echo
    CUDA_VISIBLE_DEVICES="0" python train.py \
    --expt=hid_14_emb_6_timear/run_${i} --dataset='tourism' --random_seed=${i} \
    --fixed_lstm_hidden=14 --node_emb_dim=6 --emb_reg_weight=7.2498e-8 \
    --nmf_rank=6 --dec_hid=12 --add_dec_hid=False \
    --batch_size=512 --train_epochs=40 --patience=10 --learning_rate=0.0676 --num_changes=6 \
    --hist_len=24 --pred_len=4 --val_windows=3 --test_windows=3
done
