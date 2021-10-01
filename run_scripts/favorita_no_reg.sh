set -e

for i in {1..10}
do
    echo
    echo "RUN ${i}"
    echo
    CUDA_VISIBLE_DEVICES="1" python train.py \
    --expt=hid_24_emb_8_timear_noreg/run_${i} --dataset='favorita' --random_seed=${i} \
    --fixed_lstm_hidden=24 --node_emb_dim=8 --emb_reg_weight=0.0 \
    --nmf_rank=4 --dec_hid=16 \
    --batch_size=512 --train_epochs=40 --patience=10 --learning_rate=0.002298 --num_changes=6 \
    --hist_len=28 --pred_len=7 --val_windows=5 --test_windows=5
done

# (24, 16), (45, 8), (56, 4)