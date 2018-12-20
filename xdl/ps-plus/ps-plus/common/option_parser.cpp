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

#include "ps-plus/common/option_parser.h"

#include <iostream>
#include <limits>  
#include <string.h>  

#include "ps-plus/common/string_utils.h"

using namespace std;

namespace ps {

CommandLineParameter::CommandLineParameter(const string& cmd) {
    vector<string> st = StringUtils::split(cmd, " ");
    _argc = st.size();;
    _argv = new char*[_argc];
    for (int32_t i = 0; i < _argc; ++i) {
        int32_t size = st[i].size() + 1;
        _argv[i] = new char[size];
        strncpy(_argv[i], st[i].c_str(), size);
    }
}

CommandLineParameter::~CommandLineParameter() { 
    for (int32_t i = 0; i < _argc; ++i) {
        delete[] _argv[i];
        _argv[i] = NULL;
    }
    delete[] _argv;
    _argv = NULL;
}

OptionParser::OptionParser(const string& usageDescription) { 
    _usageDescription = usageDescription;
}

OptionParser::~OptionParser() { 
}

void OptionParser::addOptionInfo(const OptionInfo &optionInfo, 
                                 const string& shortOpt,
                                 const string& longOpt) 
{
    size_t index = _optionInfos.size();
    _shortOpt2InfoMap.insert(make_pair(shortOpt, index));
    _longOpt2InfoMap.insert(make_pair(longOpt, index));
    _optionInfos.push_back(optionInfo);
}

void OptionParser::addOption(const string &shortOpt, const string &longOpt, 
               const string& optName, const char* defaultValue)
{
    addOption(shortOpt, longOpt, optName, string(defaultValue));
}

void OptionParser::addOption(const string &shortOpt, const string &longOpt, 
               const string& optName, const string& defaultValue)
{
    OptionInfo optionInfo(optName, OPT_STRING, STORE, false);
    addOptionInfo(optionInfo, shortOpt, longOpt);
    _strOptMap[optName] = defaultValue;
}

void OptionParser::addOption(const string &shortOpt, const string &longOpt, 
               const string& optName, const uint32_t defaultValue) 
{
    string defaultStringValue = StringUtils::toString(defaultValue);
    OptionInfo optionInfo(optName, OPT_UINT32, STORE, false);
    addOptionInfo(optionInfo, shortOpt, longOpt);
    _intOptMap[optName] = (int32_t)defaultValue;
}

void OptionParser::addOption(const string &shortOpt, const string &longOpt, 
               const string& optName, const int32_t defaultValue)
{
    string defaultStringValue = StringUtils::toString(defaultValue);
    OptionInfo optionInfo(optName, OPT_INT32, STORE, false);
    addOptionInfo(optionInfo, shortOpt, longOpt);
    _intOptMap[optName] = defaultValue;
}

void OptionParser::addOption(const string &shortOpt, const string &longOpt, 
               const string& optName, const bool defaultValue)
{
    string defaultStringValue = StringUtils::toString(defaultValue);
    OptionInfo optionInfo(optName, OPT_BOOL, STORE_TRUE, false);
    addOptionInfo(optionInfo, shortOpt, longOpt);
    _boolOptMap[optName] = defaultValue;
}

void OptionParser::addOption(const string& shortOpt, const string &longOpt, 
                             const string& optName, const OptionType type,
                             bool isRequired)
{
    OptionInfo optionInfo(optName, type, STORE, isRequired);
    addOptionInfo(optionInfo, shortOpt, longOpt);
}

void OptionParser::addOption(const string &shortOpt, const string &longOpt, 
                             const string& optName, const OptionAction &action,
                             const OptionType type, bool isRequired)
{
    OptionInfo optionInfo(optName, type, action, isRequired);
    addOptionInfo(optionInfo, shortOpt, longOpt);
}

bool OptionParser::parseArgs(int argc, char **argv) {
    string option;
    for (int i = 0; i < argc; ++i)
    {
        string optName = string(argv[i]);
        size_t index;
        ShortOpt2InfoMap::const_iterator shortIt = _shortOpt2InfoMap.find(optName);
        if (shortIt == _shortOpt2InfoMap.end()) {
            LongOpt2InfoMap::const_iterator longIt = _longOpt2InfoMap.find(optName);
            if (longIt == _longOpt2InfoMap.end()) {
                continue;
            } else {
                index = longIt->second;
            }
        } else {
            index = shortIt->second;
        }
        OptionInfo &info = _optionInfos[index];
        if (info.optionType == OPT_HELP) {
            cout<<_usageDescription<<endl;
            return false;
        }

        string optarg;
        if (info.hasValue()) {
            
            if (i + 1 >= argc) 
            {
                fprintf(stderr, "Option parse error: option [%s] must have value!\n", 
                        optName.c_str());
                return false;
            } else {
                ShortOpt2InfoMap::const_iterator shortIt = _shortOpt2InfoMap.find(argv[i+1]);
                LongOpt2InfoMap::const_iterator longIt = _longOpt2InfoMap.find(argv[i+1]);
                if (shortIt != _shortOpt2InfoMap.end() || (longIt != _longOpt2InfoMap.end()))
                {
                    fprintf(stderr, "Option parse error: option [%s] must have value!\n", 
                            optName.c_str());
                    return false;
                }
                optarg = argv[++i];
            }
        }

        info.isSet = true;
        switch(info.optionType) {
        case OPT_STRING:
            _strOptMap[info.optionName] = optarg;
            break;
        case OPT_INT32:
        {
            int32_t intValue;
            if (StringUtils::strToInt32(optarg.c_str(), intValue) == false) {
                fprintf(stderr, "Option parse error: invalid int32 value[%s] for option [%s]\n", 
                        optarg.c_str(), optName.c_str());
                return false;
            } 
            _intOptMap[info.optionName] = intValue;
            break;
        }
        case OPT_UINT32:
        {
            int64_t intValue;
            if (StringUtils::strToInt64(optarg.c_str(), intValue) == false
                || intValue < numeric_limits<uint32_t>::min() 
                    || intValue > numeric_limits<uint32_t>::max()) 
            {
                fprintf(stderr, "Option parse error: invalid uint32 value[%s] for option [%s]\n", 
                        optarg.c_str(), optName.c_str());
                return false;
            } 
            _intOptMap[info.optionName] = (int32_t)intValue;
            break;
        }
        case OPT_BOOL:
            _boolOptMap[info.optionName] = info.optionAction == STORE_TRUE ? true : false;
            break;
        default:
            continue;
        }
    }

    if (!isValidArgs()) {
        return false;
    }
    return true;
}

bool OptionParser::isValidArgs() {
    for (OptionInfos::const_iterator it = _optionInfos.begin();
         it != _optionInfos.end(); ++it) 
    {
        OptionInfo info = (*it);
        if (info.isRequired && info.isSet == false) {
            fprintf(stderr, "Option parse error: missing required option [%s]\n", 
                    info.optionName.c_str());
            return false;
        }
    }

    return true;
}

bool OptionParser::parseArgs(const string &commandString) {
    CommandLineParameter cmdLinePara(commandString);
    return parseArgs(cmdLinePara.getArgc(), cmdLinePara.getArgv());
}

OptionParser::StrOptMap OptionParser::getOptionValues() const {
    return _strOptMap;
}

bool OptionParser::getOptionValue(const string &optName, string &value) const {
    StrOptMap::const_iterator it = _strOptMap.find(optName);
    if (it != _strOptMap.end()) {
        value = it->second;
        return true;
    }
    return false;
}

bool OptionParser::getOptionValue(const string &optName, bool &value) const{
    BoolOptMap::const_iterator it = _boolOptMap.find(optName);
    if (it != _boolOptMap.end()) {
        value = it->second;
        return true;
    }
    return false;
}


bool OptionParser::getOptionValue(const string &optName, int32_t &value) const{
    IntOptMap::const_iterator it = _intOptMap.find(optName);
    if (it != _intOptMap.end()) {
        value = it->second;
        return true;
    }
    return false;
}

bool OptionParser::getOptionValue(const string &optName, uint32_t &value) const{
    IntOptMap::const_iterator it = _intOptMap.find(optName);
    if (it != _intOptMap.end()) {
        value = (uint32_t)it->second;
        return true;
    }
    return false;
}

} //ps
