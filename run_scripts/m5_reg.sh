set -e

# for i in {1..10}
# do
#     echo
#     echo "RUN ${i}"
#     echo
#     CUDA_VISIBLE_DEVICES="2" python train.py \
#     --expt=hid_16_exp_10_init/run_${i} --model=fixed --random_seed=${i} \
#     --fixed_lstm_hidden=16 --node_emb_dim=10 \
#     --batch_size=500 --train_epochs=25 --patience=5 --learning_rate=0.01
#     # break  # Comment when running multiple expts
# done

# for i in {1..10}
# do
#     echo
#     echo "RUN ${i}"
#     echo
#     CUDA_VISIBLE_DEVICES="3" python train.py \
#     --expt=hid_16_exp_10_ar_1e-9_er_1e-3/run_${i} --dataset='m5' --random_seed=${i} \
#     --fixed_lstm_hidden=16 --node_emb_dim=10 --act_reg_weight=1e-9 --emb_reg_weight=1e-3 \
#     --batch_size=500 --train_epochs=30 --patience=10 --learning_rate=0.005 --num_changes=8 \
#     --hist_len=28 --train_pred=28 --test_pred=7 --val_windows=5 --test_windows=5
#     # break  # Comment when running multiple expts
# done

set -e

for i in {1..10}
do
    echo
    echo "RUN ${i}"
    echo
    CUDA_VISIBLE_DEVICES="3" python train.py \
    --expt=hid_20_exp_8_reg_norm/run_${i} --dataset='m5' --random_seed=${i} \
    --fixed_lstm_hidden=20 --node_emb_dim=8 --act_reg_weight=5.5438e-6 --emb_reg_weight=1.2423e-6 \
    --loc_reg=1e-10 --local_model=True \
    --batch_size=500 --train_epochs=40 --patience=10 --learning_rate=0.01606 --num_changes=6 \
    --hist_len=28 --train_pred=28 --test_pred=7 --val_windows=5 --test_windows=5
done