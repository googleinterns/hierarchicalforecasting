
python generate_data.py \
    --random_seed=0 --model=fixed --hierarchy=sibling_reg \
    --batch_size=200 --l2_reg_weight=0.0 --l2_weight_slack=0.0 \
    --l1_reg_weight=0.0 --node_emb_dim=16 --fixed_lstm_hidden=4 \
    --overparam=False --output_scaling=False --emb_as_inp=True \
    --emb_seed=0
