set -e

for i in {1..10}
do
    CUDA_VISIBLE_DEVICES='2' python reconcile.py \
    --dir=logs/favorita/hid_64_exp_1_norm/run_${i} \
    --dataset=favorita --lamda=100.0
done

# for i in {1..10}
# do
#     CUDA_VISIBLE_DEVICES='3' python reconcile.py \
#     --dir=logs/m5/hid_64_exp_1_norm/run_${i} \
#     --dataset=m5 --lamda=0.5
# done

# for i in {1..10}
# do
#     CUDA_VISIBLE_DEVICES='2' python reconcile.py \
#     --dir=logs/tourism/hid_30_exp_1_norm/run_${i} \
#     --dataset=tourism --lamda=0.01
# done
