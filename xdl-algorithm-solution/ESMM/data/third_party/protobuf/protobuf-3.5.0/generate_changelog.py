# Copyright (C) 2016-2018 Alibaba Group Holding Limited
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python

"""Generates a friendly list of changes per language since the last release."""

import sys
import os

class Language(object):
  def __init__(self, name, pathspec):
    self.name = name
    self.pathspec = pathspec

languages = [
  Language("C++", [
      "':(glob)src/google/protobuf/*'",
      "src/google/protobuf/compiler/cpp",
      "src/google/protobuf/io",
      "src/google/protobuf/util",
      "src/google/protobuf/stubs",
  ]),
  Language("Java", [
      "java",
      "javanano",
      "src/google/protobuf/compiler/cpp",
  ]),
  Language("Python", [
      "javanano",
      "src/google/protobuf/compiler/python",
  ]),
  Language("JavaScript", [
      "js",
      "src/google/protobuf/compiler/js",
  ]),
  Language("PHP", [
      "php",
      "src/google/protobuf/compiler/php",
  ]),
  Language("Ruby", [
      "ruby",
      "src/google/protobuf/compiler/ruby",
  ]),
  Language("Csharp", [
      "csharp",
      "src/google/protobuf/compiler/csharp",
  ]),
  Language("Objective C", [
      "objectivec",
      "src/google/protobuf/compiler/objectivec",
  ]),
]

if len(sys.argv) < 2:
  print("Usage: generate_changelog.py <previous release>")
  sys.exit(1)

previous = sys.argv[1]

for language in languages:
  print(language.name)
  sys.stdout.flush()
  os.system(("git log --pretty=oneline --abbrev-commit %s...HEAD %s | " +
             "sed -e 's/^/ - /'") % (previous, " ".join(language.pathspec)))
  print("")

print("To view a commit on GitHub: " +
      "https://github.com/google/protobuf/commit/<commit id>")
