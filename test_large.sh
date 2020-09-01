python test_large.py --use_gt=False --just_test=True --valid_size=10 --train_size=10 --epoch_num=1 \
    --mix_data=False \
    --cuda=4 \
    --note=decoder-test-20 \
    --input_type=bot \
    --reward_type=pyrm-hard \
    --allow_rot=True \
    --obj_dim=2 \
    --layers=1 \
    --min_size=1 \
    --max_size=5 \
    --container_width=5 \
    --initial_container_width=8 \
    --nodes=10 \
    --total_blocks_num=20 \
    --use_heightmap=True \
    --checkpoint=./pack/old_10/2d-bot-pyrm-hard-width-5-note-decoder-new-dataset-2020-01-07-10-50/checkpoints/99
    # --checkpoint=./pack/10/2d-bot-pyrm-hard-width-5-note-normal-new-dataset-2020-01-07-10-50/checkpoints/99