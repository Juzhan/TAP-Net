python rolling.py \
    --use_cuda=True \
    --cuda=2 \
    --note=rolling_newdiff_50 \
    \
    --just_generate=False \
    --just_test=True \
    --valid_size=10000 \
    --train_size=10 \
    --epoch_num=1 \
    --mix_data=False \
    --gt_data=False \
    \
    --nodes=10 \
    --total_blocks_num=50 \
    \
    --input_type=bot \
    --reward_type=C+P+S-lb-soft \
    --packing_strategy=LB_GREEDY \
    --allow_rot=True \
    --obj_dim=2 \
    --layers=1 \
    \
    --unit=1 \
    --min_size=1 \
    --max_size=5 \
    --container_width=5 \
    --container_height=250 \
    --initial_container_width=7 \
    --initial_container_height=250 \
    \
    --decoder_input_type=shape_heightmap \
    --heightmap_type=diff \
    \
    --encoder_hidden=128 \
    --decoder_hidden=256 \
    \
    --checkpoint=pack/10/2d-bot-C+P+S-lb-soft-width-5-note-sh-R-newdiff_resume2-2020-05-09-14-39/checkpoints/299/
