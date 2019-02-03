#!/bin/bash

if [ ! -d build ]; then
    mkdir build
fi
set -x
pushd build

cmake ..

make -j 32 

popd
