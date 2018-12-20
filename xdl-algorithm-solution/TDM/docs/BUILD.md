# How to build

- Requirements

git, cmake, gcc/g++, swig, python

- Update submodule

```bash
git submodule update --init --recursive
```

- Make build directory in TDM/src directory and cmake .. and make

```bash
cd ${TDM_HOME}/src && mkdir build && cd build
cmake ..
make
```

- Install to distributed dir

```bash
cp -r ${TDM_HOME}/src/python/store/store/ ${TDM_HOME}/script/distributed/tdm_ub_att/
cp -r ${TDM_HOME}/src/python/dist_tree/dist_tree/ ${TDM_HOME}/script/distributed/tdm_ub_att/
cp -r ${TDM_HOME}/src/python/cluster/ ${TDM_HOME}/script/distributed/tdm_ub_att/
cp ${TDM_HOME}/src/build/tdm/lib*.so ${TDM_HOME}/script/distributed/tdm_ub_att/
```

## Note
- TDM_HOME is your root directory of this project
- You must build tdm by gcc 5.3.0 or upper, please using export CC=<gcc 5.3.0 path> and export CXX=<g++ 5.3.0 path>
