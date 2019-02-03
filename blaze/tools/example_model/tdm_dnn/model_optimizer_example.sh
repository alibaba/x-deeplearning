#!/bin/sh
python ../../model_optimizer.py \
  -i ./output/model_blaze \
  -o ./output/model_blaze_optimized \
  -b 1
