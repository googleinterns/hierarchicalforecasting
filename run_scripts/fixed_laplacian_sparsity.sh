set -e

for i in {1..10}
do
    echo
    echo "RUN ${i}"
    echo
    python train.py \
    --expt=sparsity_input_scaling/run_${i} --model=fixed --hierarchy=laplacian \
    --batch_size=500 --laplacian_weight=0.0001 --sparsity_weight=1e-6 \
    --train_epochs=35
done
