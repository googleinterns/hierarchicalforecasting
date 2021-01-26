set -e

for i in {1..10}
do
    echo
    echo "RUN ${i}"
    echo
    CUDA_VISIBLE_DEVICES="3" python train.py \
    --expt=node_emb_24_rs_out/run_${i} --model=fixed_df --random_seed=${i} \
    --node_emb_dim=24 --fixed_lstm_hidden=24 \
    --batch_size=500 --train_epochs=25 --patience=5 --learning_rate=0.01
    # break  # Comment when running multiple expts
done
