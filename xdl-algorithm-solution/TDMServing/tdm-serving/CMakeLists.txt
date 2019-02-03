file(GLOB_RECURSE SOURCE
    "common/common_def.cpp"
    "util/conf_parser.cpp"
    "util/file_monitor.cpp"
    "util/registerer.cpp"
    "util/str_util.cpp"
    "util/timer.cpp"
    "index/index_manager.cpp"
    "index/index_unit.cpp"
    "index/index_conf.cpp"
    "index/index.cpp"
    "index/filter.cpp"
    "index/tree/node.cpp"
    "index/tree/node_map.cpp"
    "index/tree/tree.cpp"
    "index/tree/tree_def.cpp"
    "index/tree/tree_index_conf.cpp"
    "index/tree/tree_search_context.cpp"
    "index/tree/tree_searcher.cpp"
    "index/tree/tree_index.cpp"
    "model/model_manager.cpp"
    "model/model_unit.cpp"
    "model/model_conf.cpp"
    "model/model.cpp"
    "model/blaze/blaze_def.cpp"
    "model/blaze/blaze_model_conf.cpp"
    "model/blaze/blaze_model.cpp"
    "model/blaze/embedding.cpp"
    "model/blaze/proto_configure.cpp"
    "model/blaze/quick-embedding/hashtable.cpp"
    "model/blaze/quick-embedding/quick-embedding-dict.cpp"
    "model/blaze/quick-embedding/trie.cpp"
    "model/blaze/quick-embedding/version-verifier.cpp"
    "biz/filter_manager.cpp"
    "biz/filter.cpp"
    "api/search_manager.cpp"
)

set(PROTO_FILES proto/search.proto
                proto/tree.proto)
compile_proto(PROTO_HDRS PROTO_SRCS ${PROJECT_BINARY_DIR}
                                    ${PROJECT_SOURCE_DIR}/tdm-serving
                                    "${PROTO_FILES}")

list(APPEND SOURCE ${PROTO_SRCS})

add_library(tdm_serving_static STATIC ${SOURCE})

add_library(tdm_serving SHARED ${SOURCE})
target_link_libraries(tdm_serving ${DYNAMIC_LIB})

#install
install(TARGETS tdm_serving
        LIBRARY DESTINATION lib)
install(TARGETS tdm_serving_static
        ARCHIVE DESTINATION lib)
install(FILES ${PROJECT_SOURCE_DIR}/tdm-serving/api/search_manager.h
        DESTINATION include)
install(FILES ${PROJECT_BINARY_DIR}/proto/search.pb.h
        DESTINATION include)

