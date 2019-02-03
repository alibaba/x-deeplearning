/* Copyright (C) 2016-2018 Alibaba Group Holding Limited

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TEST_INDEX_TREE_TEST_UTIL_H_
#define TEST_INDEX_TREE_TEST_UTIL_H_

namespace tdm_serving {

uint32_t GetIdBySeq(uint32_t seq);
uint32_t GetLevel(uint32_t id);

bool CreateTestTreeIndex(
    const std::string& data_path,
    bool build_meta_pb_with_other_pb_caurse_failed = false,
    bool build_meta_pb_file_not_exits_caurse_parse_failed = false,
    bool build_empty_tree = false,
    bool build_one_tree_pb_with_other_pb_caurse_parse_failed = false,
    bool build_one_tree_pb_file_not_exits_caurse_parse_failed = false,
    bool build_meta_offset_count_not_alligned = false,
    bool build_init_node_structure_child_failed = false,
    bool build_init_node_structure_parent_failed = false,
    bool build_duplicate_node = false,
    bool build_tree_node_size_not_equal_meta = false);


}  // namespace tdm_serving

#endif  // TEST_INDEX_TREE_TEST_UTIL_H_
