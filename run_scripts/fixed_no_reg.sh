set -e

for i in {1..10}
do
    echo
    echo "RUN ${i}"
    echo
    CUDA_VISIBLE_DEVICES="2" python train.py \
    --expt=mean_norm/run_${i} --model=fixed --random_seed=${i} \
    --batch_size=500 --train_epochs=25 --patience=5 --learning_rate=0.01
    # break  # Comment when running multiple expts
done
