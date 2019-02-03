#!/bin/sh
set -x

FILES=('din' 'gwen' 'tdm_att' 'tdm_dnn')

lwp=`pwd`
for file in "${FILES[@]}"
do
  cd ${lwp}/$file
  sh build_qed_example.sh
  sh model_converter_example.sh
  sh model_optimizer_example.sh
  cd ../
done
