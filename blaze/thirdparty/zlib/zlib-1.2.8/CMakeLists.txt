set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -D_LARGEFILE64_SOURCE=1 -DHAVE_HIDDEN")
set(LIB_SRCS
  adler32.c
  crc32.c
  deflate.c
  infback.c
  inffast.c
  inflate.c
  inftrees.c
  trees.c
  zutil.c
  compress.c
  uncompr.c
  gzclose.c
  gzlib.c
  gzread.c
  gzwrite.c
)
add_library(zlib SHARED ${LIB_SRCS})
add_library(zlib_static STATIC ${LIB_SRCS})
