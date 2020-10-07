set -e

for i in {1..20}
do
    echo
    echo "RUN ${i}"
    echo
    python train.py \
    --expt=iemb_add/run_${i} --random_seed=${i} --model=fixed --hierarchy=sibling_reg \
    --batch_size=200 --l2_reg_weight=0.01 --l2_weight_slack=0.01 \
    --l1_reg_weight=0.0 --node_emb_dim=16 --fixed_lstm_hidden=4 \
    --overparam=False --output_scaling=False --emb_as_inp=True \
    --train_epochs=14 --learning_rate=0.05
done
