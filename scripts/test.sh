python trainer.py \
    --use_cuda=True \
    --cuda=0 \
    --note=testing \
    \
    --task=test \
    --valid_size=10000 \
    --train_size=10 \
    --epoch_num=1 \
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
    --initial_container_height=50 \
    \
    --checkpoint=./p/2d-bot-C+P+S-lb-soft-width-5-note-sh-R-diff/