set -e

# for i in {1..10}
# do
#     echo
#     echo "RUN ${i}"
#     echo
#     CUDA_VISIBLE_DEVICES="1" python train.py \
#     --expt=hid_24_exp_24/run_${i} --dataset='favorita' --random_seed=${i} \
#     --fixed_lstm_hidden=24 --node_emb_dim=24 \
#     --batch_size=500 --train_epochs=30 --patience=10 --learning_rate=0.01 \
#     --hist_len=28 --train_pred=28 --test_pred=7 --val_windows=5 --test_windows=5
#     # break  # Comment when running multiple expts
# done

for i in {1..10}
do
    echo
    echo "RUN ${i}"
    echo
    CUDA_VISIBLE_DEVICES="1" python train.py \
    --expt=hid_70_exp_8_lr_0.002/run_${i} --dataset='favorita' --random_seed=${i} \
    --fixed_lstm_hidden=70 --node_emb_dim=8 \
    --batch_size=500 --train_epochs=30 --patience=10 --learning_rate=0.002 \
    --hist_len=28 --train_pred=28 --test_pred=7 --val_windows=5 --test_windows=5
    # break  # Comment when running multiple expts
done
