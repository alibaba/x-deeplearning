set(zlib_source_dir ${PROJECT_SOURCE_DIR}/thirdparty/zlib/zlib-1.2.8)
set(ZLIB_ROOT ${zlib_source_dir})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -D_LARGEFILE64_SOURCE=1 -DHAVE_HIDDEN")

set(zlib_srcs
  ${ZLIB_ROOT}/adler32.c
  ${ZLIB_ROOT}/crc32.c
  ${ZLIB_ROOT}/deflate.c
  ${ZLIB_ROOT}/infback.c
  ${ZLIB_ROOT}/inffast.c
  ${ZLIB_ROOT}/inflate.c
  ${ZLIB_ROOT}/inftrees.c
  ${ZLIB_ROOT}/trees.c
  ${ZLIB_ROOT}/zutil.c
  ${ZLIB_ROOT}/compress.c
  ${ZLIB_ROOT}/uncompr.c
  ${ZLIB_ROOT}/gzclose.c
  ${ZLIB_ROOT}/gzlib.c
  ${ZLIB_ROOT}/gzread.c
  ${ZLIB_ROOT}/gzwrite.c
)

list(APPEND LIB_ZLIB_SOURCE ${zlib_srcs})
