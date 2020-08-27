set -e

for i in {1..10}
do
    echo
    echo "RUN ${i}"
    echo
    python train.py \
    --expt=output_scaling_trans_l2_10_l1_0.1/run_${i} --model=fixed \
    --hierarchy=add_dev \
    --batch_size=500 --l2_reg_weight=10 --l1_reg_weight=0.1 --node_emb_dim=16 \
    --overparam=True --input_scaling=False --output_scaling=True \
    --train_epochs=25 --learning_rate=0.01
done

# for ((l1=-5; l1<=-1; l1++)) do
#     for ((l2=-5; l2<=-1; l2++)) do
#         for i in {1..10}
#         do
#             echo
#             echo "L1=${l1} L2=${l2} RUN ${i}"
#             echo
#             python train.py \
#             --expt=HPO/l2_1e${l2}_l1_1e${l1}/run_${i} --model=fixed \
#             --hierarchy=add_dev \
#             --batch_size=500 --l2_reg_weight=1e${l2} --l1_reg_weight=1e${l1} \
#             --node_emb_dim=16 \
#             --overparam=True --input_scaling=False --output_scaling=True \
#             --train_epochs=25 --learning_rate=0.01
#         done
#     done
# done
