set -e

for i in {1..20}
do
    echo
    echo "RUN ${i}"
    echo
    python train.py \
    --expt=iemb_frac_0.2/run_${i} --random_seed=${i} --model=fixed --hierarchy=sibling_reg \
    --batch_size=200 --l2_reg_weight=0.0 --l2_weight_slack=0.0 \
    --l1_reg_weight=0.0 --node_emb_dim=16 --fixed_lstm_hidden=16 \
    --overparam=False --output_scaling=False --emb_as_inp=True \
    --train_epochs=20 --learning_rate=0.005 --data_frac=0.2
done

# --hierarchy=sibling_reg

# python train.py \
#     --expt=false_data_iemb/run_0 --random_seed=0 --model=fixed --hierarchy=sibling_reg \
#     --batch_size=200 --l2_reg_weight=0.0 --l2_weight_slack=0.0 \
#     --l1_reg_weight=0.0 --node_emb_dim=16 --fixed_lstm_hidden=4 \
#     --overparam=False --output_scaling=False --emb_as_inp=True \
#     --train_epochs=14 --learning_rate=0.05
