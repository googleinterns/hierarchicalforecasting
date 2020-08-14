set -e

for i in {1..10}
do
    echo
    echo "RUN ${i}"
    echo
    python train.py \
    --expt=te_35/run_${i} --model=fixed \
    --batch_size=500 --train_epochs=35
done
