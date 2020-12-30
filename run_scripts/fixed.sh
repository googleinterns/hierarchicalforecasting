CUDA_VISIBLE_DEVICES='3' python train.py \
    --expt=syn_factors --random_seed=0 --model=fixed \
    --data_fraction=1.0 \
    --batch_size=100 --l2_reg_weight=1.0 --l2_weight_slack=0.0 \
    --l1_reg_weight=0.0 --node_emb_dim=10 --fixed_lstm_hidden=10 \
    --overparam=False --output_scaling=False --emb_as_inp=False \
    --train_epochs=100 --learning_rate=0.01 --patience=10
