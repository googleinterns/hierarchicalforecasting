# set -e

# for i in {1..20}
# do
#     echo
#     echo "RUN ${i}"
#     echo
#     python train.py \
#     --expt=false_data_iemb/run_${i} --random_seed=${i} --model=fixed \
#     --batch_size=200 --l2_reg_weight=0.0 --l2_weight_slack=0.0 \
#     --l1_reg_weight=0.0 --node_emb_dim=16 --fixed_lstm_hidden=4 \
#     --overparam=False --output_scaling=False --emb_as_inp=True \
#     --train_epochs=14 --learning_rate=0.05 --load_alternate=True
# done
# --hierarchy=sibling_reg

CUDA_VISIBLE_DEVICES='3' python train.py \
    --expt=syn_iemb --random_seed=0 --model=fixed \
    --data_fraction=10.0 --hierarchy=add_dev \
    --batch_size=200 --l2_reg_weight=1.0 --l2_weight_slack=0.0 \
    --l1_reg_weight=0.0 --node_emb_dim=10 --fixed_lstm_hidden=10 \
    --overparam=False --output_scaling=False --emb_as_inp=True \
    --train_epochs=30 --learning_rate=0.01
