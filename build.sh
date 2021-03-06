#!/usr/bin/env sh
GP_DIR=$(pwd)
DEPS_DIR=$(pwd)/deps

mkdir -p $DEPS_DIR

# get alglib and eigen (for the gaussian process library)
cd $DEPS_DIR
if [ ! -d "alglib" ]; then
    mkdir -p $DEPS_DIR/alglib
    echo "Downloading ALGLIB..."
    wget -O $DEPS_DIR/alglib.tgz http://www.alglib.net/translator/re/alglib-3.8.2.cpp.tgz
    tar xvzf $DEPS_DIR/alglib.tgz -C $DEPS_DIR/alglib
    echo "Patching ALGLIB..."
    # apply patch to alglib
    patch $DEPS_DIR/alglib/cpp/src/optimization.h $DEPS_DIR/../alglib_patches/optimization.h.patch
    patch $DEPS_DIR/alglib/cpp/src/optimization.cpp $DEPS_DIR/../alglib_patches/optimization.cpp.patch
fi

if [ ! -d "eigen" ]; then
    echo "Getting libeigen3..."
    hg clone https://bitbucket.org/eigen/eigen/
fi
#sudo apt-get install libeigen3-dev
cd $GP_DIR

# update links to dependencies
rm -f ./include/alglib ./include/eigen
ln -s $DEPS_DIR/alglib/cpp ./include/alglib
ln -s $DEPS_DIR/eigen ./include/eigen

# build
mkdir -p build
cd build
cmake ..
make -j3
cd ..
