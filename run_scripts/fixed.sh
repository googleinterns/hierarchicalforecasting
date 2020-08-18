set -e

for i in {1..10}
do
    echo
    echo "RUN ${i}"
    echo
    python train.py \
    --expt=trans_no_scaling_sgd/run_${i} --model=fixed \
    --batch_size=500 --train_epochs=25 --learning_rate=0.01
done
