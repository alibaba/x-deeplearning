#!/bin/sh
mkdir -p output

python ../../build_qed.py \
  -p ./model/ \
  -i sparse.txt \
  -o ./output/sparse_qed \
  -s 0
