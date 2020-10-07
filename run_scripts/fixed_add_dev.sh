set -e

for i in {1..20}
do
    echo
    echo "RUN ${i}"
    echo
    python train.py \
    --expt=opt_hp/run_${i} --random_seed=${i} --model=fixed --hierarchy=add_dev \
    --batch_size=500 --l2_reg_weight=1.0 --l1_reg_weight=0.0 --node_emb_dim=16 \
    --fixed_lstm_hidden=4 \
    --overparam=True --output_scaling=True \
    --train_epochs=14 --learning_rate=0.05
done
