set -e

for i in {1..10}
do
    echo
    echo "RUN ${i}"
    echo
    python train.py \
    --expt=input_scaling_positive/run_${i} --model=fixed --hierarchy=laplacian \
    --batch_size=500 --laplacian_weight=0.00005 --train_epochs=35
done
