set -e

for i in {1..10}
do
    echo
    echo "RUN ${i}"
    echo
    python train.py \
    --expt=output_scaling_trans_l2_norm/run_${i} --model=fixed --hierarchy=add_dev \
    --batch_size=500 --sparsity_weight=1.0 --node_emb_dim=16 \
    --train_epochs=25 --learning_rate=0.01
done
