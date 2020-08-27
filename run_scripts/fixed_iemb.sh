set -e

for i in {1..10}
do
    echo
    echo "RUN ${i}"
    echo
    python train.py \
    --expt=iemb_no_oemb_l2_0_l1_0/run_${i} --model=fixed \
    --batch_size=500 --l2_reg_weight=0 --l1_reg_weight=0 --node_emb_dim=16 \
    --overparam=True --input_scaling=False --output_scaling=True --emb_as_inp=True \
    --train_epochs=25 --learning_rate=0.01
done

# --hierarchy=add_dev
