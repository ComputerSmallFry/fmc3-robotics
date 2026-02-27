#!/bin/bash
python ~/workspace/dataset/fourier/convert_tools/convert_dora_to_lerobot.py \
    --input ~/workspace/dataset/fourier/dora-record/019c507b-7fcb-7ccf-869a-abe0333177a2 \
    --output ~/workspace/dataset/fourier/pick_bottle \
    --task "grab the bottle on the table" \
    --fps 30 \
    --robot-type fourier_gr2 \
    --video-codec libopenh264
