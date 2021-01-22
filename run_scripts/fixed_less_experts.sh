set -e

# for i in {1..10}
# do
#     echo
#     echo "RUN ${i}"
#     echo
#     CUDA_VISIBLE_DEVICES="1" python train.py \
#     --expt=exp_2_lr_0.01_ep_25/run_${i} --model=fixed --random_seed=${i} \
#     --fixed_lstm_hidden=45 --node_emb_dim=2 \
#     --batch_size=500 --train_epochs=25 --patience=5 --learning_rate=0.01
#     # break  # Comment when running multiple expts
# done

# for i in {1..10}
# do
#     echo
#     echo "RUN ${i}"
#     echo
#     CUDA_VISIBLE_DEVICES="1" python train.py \
#     --expt=exp_1_lr_0.01_ep_25/run_${i} --model=fixed --random_seed=${i} \
#     --fixed_lstm_hidden=64 --node_emb_dim=1 \
#     --batch_size=500 --train_epochs=25 --patience=5 --learning_rate=0.01
#     # break  # Comment when running multiple expts
# done

# for i in {1..10}
# do
#     echo
#     echo "RUN ${i}"
#     echo
#     CUDA_VISIBLE_DEVICES="1" python train.py \
#     --expt=exp_4_lr_0.01_ep_25/run_${i} --model=fixed --random_seed=${i} \
#     --fixed_lstm_hidden=32 --node_emb_dim=4 \
#     --batch_size=500 --train_epochs=25 --patience=5 --learning_rate=0.01
#     # break  # Comment when running multiple expts
# done

for i in {1..10}
do
    echo
    echo "RUN ${i}"
    echo
    CUDA_VISIBLE_DEVICES="1" python train.py \
    --expt=exp_32_lr_0.01_ep_25/run_${i} --model=fixed --random_seed=${i} \
    --fixed_lstm_hidden=11 --node_emb_dim=32 \
    --batch_size=500 --train_epochs=25 --patience=5 --learning_rate=0.01
    # break  # Comment when running multiple expts
done
