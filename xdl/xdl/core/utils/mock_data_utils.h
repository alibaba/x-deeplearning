/*
 * Copyright 1999-2017 Alibaba Group.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#ifndef XDL_CORE_UTILS_MOCK_DATA_UTILS_H_
#define XDL_CORE_UTILS_MOCK_DATA_UTILS_H_

#include <algorithm>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace xdl {

template <typename T>
std::string Encode(const std::vector<T>& data) {
  std::stringstream ss;
  std::copy(data.begin(), data.end(), std::ostream_iterator<T>(ss, " "));
  return ss.str();
}

template <typename T>
std::vector<T> Decode(const std::string& data) {
  std::stringstream ss(data);
  std::vector<T> res;
  std::copy(std::istream_iterator<T>(ss), std::istream_iterator<T>(),
            std::back_inserter(res));
  return res;
}

}  // namespace xdl

#endif  // XDL_CORE_UTILS_MOCK_DATA_UTILS_H_
