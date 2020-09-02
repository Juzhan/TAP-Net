python trainer.py \
    --cuda=0 \
    --note=training \
    \
    --task=train \
    --valid_size=10 \
    --train_size=128000 \
    --epoch_num=200 \
    --dataset=RAND \
    \
    --nodes=10 \
    --obj_dim=2 \
    \
    --unit=1 \
    --arm_size=1 \
    --min_size=1 \
    --max_size=5 \
    --container_width=5 \
    --container_height=50 \
    --initial_container_width=7 \
    --initial_container_height=50