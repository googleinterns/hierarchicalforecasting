set -e

for i in {1..10}
do
    echo
    echo "RUN ${i}"
    echo
    python train.py \
    --expt=te_35/run_${i} --model=fixed --hierarchy=laplacian \
    --batch_size=500 --laplacian_weight=0.0005 --train_epochs=35
done
