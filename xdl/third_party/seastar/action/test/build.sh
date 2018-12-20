#!/bin/bash

BUILD_SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})
ACTION_SOURCE_DIR=$(cd -P ${BUILD_SCRIPT_DIR}/.. && pwd)
SEASTAR_SOURCE_DIR=$(cd -P ${ACTION_SOURCE_DIR}/.. && pwd)
SERVICE_SOURCE_DIR=$(cd -P ${SEASTAR_SOURCE_DIR}/service && pwd)

USER_CFLAGS=""
USER_CFLAGS="${USER_CFLAGS} -I/usr/local/cryptopp-5.6.5/include"
USER_CFLAGS="${USER_CFLAGS} -I/usr/local/include"
USER_CFLAGS="${USER_CFLAGS} -I/usr/include"
USER_CFLAGS="${USER_CFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0"

USER_LDFLAGS=""
USER_LDFLAGS="${USER_LDFLAGS} -L/usr/local/cryptopp-5.6.5/lib"
USER_LDFLAGS="${USER_LDFLAGS} -L/usr/local/lib64/boost"
USER_LDFLAGS="${USER_LDFLAGS} -L/usr/lib64"
USER_LDFLAGS="${USER_LDFLAGS} -Wl,-rpath,/usr/local/lib64/boost"
USER_LDFLAGS="${USER_LDFLAGS} -lboost_timer"
USER_LDFLAGS="${USER_LDFLAGS} -lboost_chrono"

pushd ${SEASTAR_SOURCE_DIR} &&                           \
./configure.py --mode=release --embedded-static          \
               --cflags="${USER_CFLAGS}"                 \
               --ldflags="${USER_LDFLAGS}" &&            \
ninja-build -j16 &&                                       \
popd &&                                                  \
cmake -DCMAKE_BUILD_TYPE=Release                         \
      -DCMAKE_C_COMPILER=/usr/local/gcc-5.3.0/bin/gcc    \
      -DCMAKE_CXX_COMPILER=/usr/local/gcc-5.3.0/bin/g++  \
      ${ACTION_SOURCE_DIR} &&                           \
make -j8 VERBOSE=1 &&                                    \
echo OK
