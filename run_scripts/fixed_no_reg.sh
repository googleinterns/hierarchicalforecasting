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
    --expt=hid_16_exp_10_pos_emb/run_${i} --model=fixed --random_seed=${i} \
    --fixed_lstm_hidden=16 --node_emb_dim=10 \
    --batch_size=500 --train_epochs=25 --patience=5 --learning_rate=0.01
    # break  # Comment when running multiple expts
done
