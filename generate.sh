python trainer.py \
    \
    --seed=12345 \
    \
    --task=generate \
    --valid_size=10000 \
    --train_size=64000 \
    --epoch_num=1 \
    --dataset=RAND \
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
    --max_size=5 \
    --container_width=5 \
    --container_height=50 \
    --initial_container_width=7 \
    --initial_container_height=50