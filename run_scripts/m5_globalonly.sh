set -e

# for i in {1..10}
# do
#     echo
#     echo "RUN ${i}"
#     echo
#     CUDA_VISIBLE_DEVICES="1" python train.py \
#     --expt=hid_42_emb_8_timear_globalonly/run_${i} --dataset='m5' --random_seed=${i} \
#     --fixed_lstm_hidden=42 --node_emb_dim=8 --emb_reg_weight=0.0 \
#     --nmf_rank=12 --dec_hid=24 --ar_ablation=True \
#     --batch_size=512 --train_epochs=40 --patience=10 --learning_rate=0.004 --num_changes=6 \
#     --hist_len=28 --pred_len=7 --val_windows=5 --test_windows=5
# done

for i in {1..10}
do
    echo
    echo "RUN ${i}"
    echo
    CUDA_VISIBLE_DEVICES="2" python train.py \
    --expt=hid_42_emb_8_timear_globalregonly/run_${i} --dataset='m5' --random_seed=${i} \
    --fixed_lstm_hidden=42 --node_emb_dim=8 --emb_reg_weight=3.414e-6 \
    --nmf_rank=12 --dec_hid=24 --ar_ablation=True \
    --batch_size=512 --train_epochs=40 --patience=10 --learning_rate=0.004 --num_changes=6 \
    --hist_len=28 --pred_len=7 --val_windows=5 --test_windows=5
done
