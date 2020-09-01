python trainer.py \
    \
    --seed=12345 \
    \
    --task=generate \
    --valid_size=300 \
    --train_size=10 \
    --epoch_num=1 \
    --dataset=PPSG \
    \
    --nodes=10 \
    \
    --input_type=bot \
    --reward_type=C+P+S-lb-soft \
    --packing_strategy=LB_GREEDY \
    --allow_rot=True \
    --obj_dim=2 \
    --layers=1 \
    \
    --unit=1 \
    --arm_size=1 \
    --min_size=1 \
    --max_size=50 \
    --container_width=50 \
    --container_height=500 \
    --initial_container_width=70 \
    --initial_container_height=500