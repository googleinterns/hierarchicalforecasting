set -e

for i in {1..10}
do
    echo
    echo "Favorita RUN ${i}"
    echo
    CUDA_VISIBLE_DEVICES="1" python train.py \
    --expt=hid_64_emb_8_reg/run_${i} --dataset='favorita' --random_seed=${i} \
    --node_emb_dim=8 --fixed_lstm_hidden=64 --emb_reg_weight=4.921e-4 \
    --batch_size=500 --train_epochs=40 --patience=10 --learning_rate=0.01 --num_changes=6 \
    --hist_len=28 --train_pred=28 --test_pred=7 --val_windows=5 --test_windows=5
done

for i in {1..10}
do
    echo
    echo "M5 RUN ${i}"
    echo
    CUDA_VISIBLE_DEVICES="1" python train.py \
    --expt=hid_64_emb_8_reg/run_${i} --dataset='m5' --random_seed=${i} \
    --node_emb_dim=8 --fixed_lstm_hidden=64 --emb_reg_weight=1.2423e-6 \
    --batch_size=500 --train_epochs=40 --patience=10 --learning_rate=0.01 --num_changes=6 \
    --hist_len=28 --train_pred=28 --test_pred=7 --val_windows=5 --test_windows=5
done

for i in {1..10}
do
    echo
    echo "Tourism RUN ${i}"
    echo
    CUDA_VISIBLE_DEVICES="1" python train.py \
    --expt=hid_20_exp_4_reg/run_${i} --dataset='tourism' --random_seed=${i} \
    --fixed_lstm_hidden=20 --node_emb_dim=4 --emb_reg_weight=3.027e-9 \
    --batch_size=512 --train_epochs=40 --patience=10 --learning_rate=0.1 --num_changes=6 \
    --hist_len=24 --train_pred=4 --test_pred=4 --val_windows=3 --test_windows=3
done
