set -e

for i in {1..20}
do
    echo
    echo "RUN ${i}"
    echo
    python train.py \
    --expt=run_${i} --random_seed=${i} --model=fixed \
    --batch_size=500 --node_emb_dim=16 --lstm_hidden_dim=64 \
    --overparam=True --output_scaling=False \
    --train_epochs=30 --learning_rate=0.01
done
