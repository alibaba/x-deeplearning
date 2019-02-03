#!/bin/sh
# convert raw xdl model
mkdir -p output

# convert xdl model with ulf
python ../../model_converter.py \
  -c ./model/graph_ulf.txt \
  -d ./model/dense.txt \
  -o ./output/model_blaze \
  -b 1

