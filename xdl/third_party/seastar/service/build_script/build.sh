#!/bin/bash

BUILD_SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})
SERVICE_SOURCE_DIR=$(cd -P ${BUILD_SCRIPT_DIR}/.. && pwd)
SEASTAR_SOURCE_DIR=$(cd -P ${SERVICE_SOURCE_DIR}/.. && pwd)

# trick for: force to rebuild libps_network_shared.so/a everytime
# `make clean`

USER_CFLAGS=""
USER_CFLAGS="${USER_CFLAGS} -I/usr/local/include"
USER_CFLAGS="${USER_CFLAGS} -I/usr/include"
USER_CFLAGS="${USER_CFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0"

USER_LDFLAGS=""
USER_LDFLAGS="${USER_LDFLAGS} -L/usr/local/lib64/boost"
USER_LDFLAGS="${USER_LDFLAGS} -lboost_timer"
USER_LDFLAGS="${USER_LDFLAGS} -lboost_chrono"
USER_LDFLAGS="${USER_LDFLAGS} -lboost_program_options"
#USER_LDFLAGS="${USER_LDFLAGS} -ljemalloc"

pushd ${SEASTAR_SOURCE_DIR} &&                           \
./configure.py --mode=release --embedded-static          \
               --cflags="${USER_CFLAGS}"                 \
               --ldflags="${USER_LDFLAGS}" &&            \
ninja -j8 &&                                       \
popd &&                                                  \
#cmake -DUSE_STATISTICS=1 \
cmake -DCMAKE_BUILD_TYPE=Release                         \
      ${SERVICE_SOURCE_DIR} &&                           \
make -j32 VERBOSE=1 &&                                   \
echo OK
