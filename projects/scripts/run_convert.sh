#!/bin/bash
python convert_dora_to_lerobot.py \
    --input ./dora-record/019c507b-7fcb-7ccf-869a-abe0333177a2 \
    --output ./pick_and_place \
    --task "grab the bottle on the table" \
    --fps 30 \
    --robot-type fourier_gr2 \
    --video-codec libopenh264
