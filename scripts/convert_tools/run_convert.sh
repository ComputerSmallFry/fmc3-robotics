#!/bin/bash
python ~/workspace/dataset/fourier/scripts/convert_tools/convert_dora_to_lerobot.py \
    --input ~/workspace/dataset/fourier/pick_bottle_and_place_into_box \
    --output ~/workspace/dataset/fourier/pick_bottle_and_place_into_box_lerobot_gr2 \
    --task "pick bottle and place into box" \
    --fps 30 \
    --robot-type fourier_gr2 \
    --video-codec libopenh264 \
    --workers 4
