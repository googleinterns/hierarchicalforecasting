set -e

for i in {1..10}
do
    echo
    echo "RUN ${i}"
    echo
    CUDA_VISIBLE_DEVICES="2" python train.py \
    --expt=hid_40_exp_4_reg_norm/run_${i} --dataset='wiki' --random_seed=${i} \
    --fixed_lstm_hidden=40 --node_emb_dim=4 --local_model=True \
    --act_reg_weight=3.758e-8 --emb_reg_weight=0.1 --loc_reg=2.9343e-9 \
    --batch_size=512 --train_epochs=40 --patience=10 --learning_rate=0.011 --num_changes=6 \
    --hist_len=28 --train_pred=28 --test_pred=7 --val_windows=5 --test_windows=5
done
