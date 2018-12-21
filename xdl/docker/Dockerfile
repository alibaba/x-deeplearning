# BASE image
FROM ubuntu:16.04

# Install envs
RUN apt-get update -y && apt-get -y upgrade; \
    apt-get install -y build-essential gcc-4.8 g++-4.8 gcc-5 g++-5 cmake python python-pip openjdk-8-jdk wget && pip install --upgrade pip; \
    cp -f /usr/local/bin/pip /usr/bin/pip; \
    cd /tmp && wget -O boost_1_63_0.tar.gz https://sourceforge.net/projects/boost/files/boost/1.63.0/boost_1_63_0.tar.gz; \
    tar zxf boost_1_63_0.tar.gz && cd boost_1_63_0; \
    ./bootstrap.sh --prefix=/usr/local && ./b2 -j32 variant=release define=_GLIBCXX_USE_CXX11_ABI=0 install; \
    mkdir -p /usr/local/lib64/boost && cp -r /usr/local/lib/libboost* /usr/local/lib64/boost/; \
    cd && rm -rf /tmp/boost_1_63_0*; \
    apt-get install -y libaio-dev ninja-build ragel libhwloc-dev libnuma-dev libpciaccess-dev libcrypto++-dev libxml2-dev xfslibs-dev libgnutls28-dev liblz4-dev libsctp-dev libprotobuf-dev protobuf-compiler libunwind8-dev systemtap-sdt-dev libjemalloc-dev libtool python3 libjsoncpp-dev; \
    cd /tmp; \
    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb; \
    wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb; \
    wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb; \
    wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl2_2.1.4-1+cuda9.0_amd64.deb; \
    wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl-dev_2.1.4-1+cuda9.0_amd64.deb; \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub; \
    dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb; \
    dpkg -i libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb; \
    dpkg -i libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb; \
    dpkg -i libnccl2_2.1.4-1+cuda9.0_amd64.deb; \
    dpkg -i libnccl-dev_2.1.4-1+cuda9.0_amd64.deb; \
    apt-get clean && apt-get update -y && apt-get -y upgrade; \
    apt-get install -y cuda=9.0.176-1 --fix-missing; \
    apt-get install -y libcudnn7-dev; \
    apt-get install -y libnccl-dev; \
    apt-get update -y && apt-get -y upgrade; \
    rm -rf /tmp/*.deb; \
    echo '/usr/local/nvidia/lib64/' >> /etc/ld.so.conf;
