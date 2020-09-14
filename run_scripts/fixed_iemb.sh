set -e

for i in {1..10}
do
    echo
    echo "RUN ${i}"
    echo
    python train.py \
    --expt=iemb_add_rerun/run_${i} --random_seed=${i} --model=fixed \
    --batch_size=500 --l2_reg_weight=0 --node_emb_dim=16 \
    --overparam=True --output_scaling=True --emb_as_inp=True \
    --train_epochs=25 --learning_rate=0.01 --hierarchy=add_dev
done
