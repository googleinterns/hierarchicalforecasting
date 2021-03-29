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
    --expt=hid_100_exp_1/run_${i} --dataset='m5' --random_seed=${i} \
    --fixed_lstm_hidden=100 --node_emb_dim=1 \
    --batch_size=500 --train_epochs=25 --patience=5 --learning_rate=0.005 --num_changes=8 \
    --hist_len=28 --train_pred=28 --test_pred=7 --val_windows=5 --test_windows=5
    # break  # Comment when running multiple expts
done
