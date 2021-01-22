set -e

# for i in {1..10}
# do
#     echo
#     echo "RUN ${i}"
#     echo
#     CUDA_VISIBLE_DEVICES="3" python train.py \
#     --expt=lr_0.01_ep_25_l1_10.0/run_${i} --model=fixed --random_seed=${i} \
#     --batch_size=500 --train_epochs=25 --patience=5 --learning_rate=0.01 \
#     --reg_type='l1' --reg_weight=10.0
#     # break  # Comment when running multiple expts
# done

for i in {1..10}
do
    echo
    echo "RUN ${i}"
    echo
    CUDA_VISIBLE_DEVICES="3" python train.py \
    --expt=exp_32_l1_0.01/run_${i} --model=fixed --random_seed=${i} \
    --batch_size=500 --train_epochs=25 --patience=5 --learning_rate=0.01 \
    --fixed_lstm_hidden=11 --node_emb_dim=32 \
    --reg_type='l1' --reg_weight=0.01
    # break  # Comment when running multiple expts
done
