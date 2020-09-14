set -e

for i in {1..10}
do
    echo
    echo "RUN ${i}"
    echo
    python train.py \
    --expt=rerun_2/run_${i} --random_seed=${i} --model=fixed --hierarchy=sibling_reg \
    --batch_size=500 --l2_reg_weight=1.0 --l2_weight_slack=1.0 --node_emb_dim=16 \
    --overparam=True --input_scaling=False --output_scaling=True \
    --train_epochs=25 --learning_rate=0.01
done
