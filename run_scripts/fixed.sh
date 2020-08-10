set -e

for i in {1..10}
do
    echo
    echo "RUN ${i}"
    echo
    python train.py \
    --expt=run_${i} --model=fixed \
    --batch_size=500
done
