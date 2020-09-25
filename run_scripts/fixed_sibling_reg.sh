set -e

for i in {1..20}
do
    echo
    echo "RUN ${i}"
    echo
    python train.py \
    --expt=opt_hps/run_${i} --random_seed=${i} --model=fixed --hierarchy=sibling_reg \
    --batch_size=350 --l2_reg_weight=0.6 --l2_weight_slack=0.1 --l1_reg_weight=0.002 \
    --node_emb_dim=18 --fixed_lstm_hidden=15 --overparam=True --output_scaling=True \
    --train_epochs=20 --learning_rate=0.01
done
