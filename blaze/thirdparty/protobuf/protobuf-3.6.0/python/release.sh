#!/bin/bash

set -ex

function get_source_version() {
  grep "__version__ = '.*'" python/google/protobuf/__init__.py | sed -r "s/__version__ = '(.*)'/\1/"
}

function run_install_test() {
  local VERSION=$1
  local PYTHON=$2
  local PYPI=$3

  virtualenv --no-site-packages -p `which $PYTHON` test-venv

  # Intentionally put a broken protoc in the path to make sure installation
  # doesn't require protoc installed.
  touch test-venv/bin/protoc
  chmod +x test-venv/bin/protoc

  source test-venv/bin/activate
  pip install -i ${PYPI} protobuf==${VERSION} --no-cache-dir
  deactivate
  rm -fr test-venv
}


[ $# -lt 1 ] && {
  echo "Usage: $0 VERSION ["
  echo ""
  echo "Examples:"
  echo "  Test 3.3.0 release using version number 3.3.0.dev1:"
  echo "    $0 3.0.0 dev1"
  echo "  Actually release 3.3.0 to PyPI:"
  echo "    $0 3.3.0"
  exit 1
}
VERSION=$1
DEV=$2

# Make sure we are in a protobuf source tree.
[ -f "python/google/protobuf/__init__.py" ] || {
  echo "This script must be ran under root of protobuf source tree."
  exit 1
}

# Make sure all files are world-readable.
find python -type d -exec chmod a+r,a+x {} +
find python -type f -exec chmod a+r {} +
umask 0022

# Check that the supplied version number matches what's inside the source code.
SOURCE_VERSION=`get_source_version`

[ "${VERSION}" == "${SOURCE_VERSION}" -o "${VERSION}.${DEV}" == "${SOURCE_VERSION}" ] || {
  echo "Version number specified on the command line ${VERSION} doesn't match"
  echo "the actual version number in the source code: ${SOURCE_VERSION}"
  exit 1
}

TESTING_ONLY=1
TESTING_VERSION=${VERSION}.${DEV}
if [ -z "${DEV}" ]; then
  read -p "You are releasing ${VERSION} to PyPI. Are you sure? [y/n]" -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi
  TESTING_ONLY=0
  TESTING_VERSION=${VERSION}
else
  # Use dev version number for testing.
  sed -i -r "s/__version__ = '.*'/__version__ = '${VERSION}.${DEV}'/" python/google/protobuf/__init__.py
fi

cd python

# Run tests locally.
python setup.py build
python setup.py test

# Deploy source package to testing PyPI
python setup.py sdist upload -r https://test.pypi.org/legacy/

# Test locally with different python versions.
run_install_test ${TESTING_VERSION} python2.7 https://test.pypi.org/simple
run_install_test ${TESTING_VERSION} python3.4 https://test.pypi.org/simple

# Deploy egg/wheel packages to testing PyPI and test again.
python setup.py bdist_egg bdist_wheel upload -r https://test.pypi.org/legacy/

run_install_test ${TESTING_VERSION} python2.7 https://test.pypi.org/simple
run_install_test ${TESTING_VERSION} python3.4 https://test.pypi.org/simple

echo "All install tests have passed using testing PyPI."

if [ $TESTING_ONLY -eq 0 ]; then
  read -p "Publish to PyPI? [y/n]" -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi
  echo "Publishing to PyPI..."
  # Be sure to run build before sdist, because otherwise sdist will not include
  # well-known types.
  python setup.py clean build sdist upload
  # Be sure to run clean before bdist_xxx, because otherwise bdist_xxx will
  # include files you may not want in the package. E.g., if you have built
  # and tested with --cpp_implemenation, bdist_xxx will include the _message.so
  # file even when you no longer pass the --cpp_implemenation flag. See:
  #   https://github.com/google/protobuf/issues/3042
  python setup.py clean build bdist_egg bdist_wheel upload
else
  # Set the version number back (i.e., remove dev suffix).
  sed -i -r "s/__version__ = '.*'/__version__ = '${VERSION}'/" google/protobuf/__init__.py
fi
