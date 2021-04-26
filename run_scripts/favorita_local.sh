set -e

for i in {1..10}
do
    echo
    echo "RUN ${i}"
    echo
    CUDA_VISIBLE_DEVICES="1" python train.py \
    --expt=hid_20_exp_8_local_ep_50/run_${i} --dataset='favorita' --random_seed=${i} \
    --fixed_lstm_hidden=20 --node_emb_dim=8 \
    --act_reg_weight=9.3e-10 --emb_reg_weight=0.0993 --local_model=True --loc_reg=2.41e-9 \
    --batch_size=500 --train_epochs=50 --patience=15 --learning_rate=0.003 --num_changes=7 \
    --hist_len=28 --train_pred=28 --test_pred=7 --val_windows=5 --test_windows=5
done
