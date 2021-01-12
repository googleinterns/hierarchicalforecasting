set -e

for i in {1..10}
do
    echo
    echo "RUN ${i}"
    echo
    CUDA_VISIBLE_DEVICES="3" python train.py \
    --expt=run_${i} --model=fixed --random_seed=${i} \
    --batch_size=500 --train_epochs=100 --patience=10 --learning_rate=0.01 \
    --reg_type='l1' --reg_weight=0.01
    break  # Comment when running multiple expts
done
