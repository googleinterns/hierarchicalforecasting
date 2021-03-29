set -e

# for i in {1..10}
# do
#     CUDA_VISIBLE_DEVICES='1' python reconcile.py \
#     --dir=logs/favorita/hid_90_exp_1/run_${i} \
#     --dataset=favorita --lamda=10.0
# done

for i in {1..10}
do
    CUDA_VISIBLE_DEVICES='2' python reconcile.py \
    --dir=logs/m5/hid_100_exp_1/run_${i} \
    --dataset=m5 --lamda=0.05
done
