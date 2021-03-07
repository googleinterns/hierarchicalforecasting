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

for i in {1..10}
do
    echo
    echo "RUN ${i}"
    echo
    CUDA_VISIBLE_DEVICES="2" python train.py \
    --expt=hid_20_exp_8_lr_0.003/run_${i} --dataset='favorita' --random_seed=${i} \
    --fixed_lstm_hidden=20 --node_emb_dim=8 \
    --batch_size=500 --train_epochs=30 --patience=10 --learning_rate=0.003 --num_changes=6 \
    --hist_len=28 --train_pred=28 --test_pred=7 --val_windows=5 --test_windows=5
    # break  # Comment when running multiple expts
done

# --expt=hid_20_exp_8_lr_0.003/run_10
#         --dataset=favorita
#         --train_epochs=30
#         --batch_size=500
#         --hist_len=28
#         --train_pred=28
#         --test_pred=7
#         --val_windows=5
#         --test_windows=5
#         --fixed_lstm_hidden=20
#         --node_emb_dim=8
#         --random_seed=10
#         --patience=10
#         --num_changes=6
#         --learning_rate=0.003
#         --emb_reg_weight=0.0
#         --act_reg_weight=0.0
