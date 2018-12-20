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

#ifndef PS_COMMON_OPTION_PARSER_H
#define PS_COMMON_OPTION_PARSER_H

#include <string>
#include <vector>
#include <map>

namespace ps {

class CommandLineParameter
{
public:
    CommandLineParameter(const std::string& cmd);
    ~CommandLineParameter();
public:
    inline int getArgc() const { return _argc; };
    inline char** getArgv() const { return _argv; };
private:
    int _argc;
    char** _argv;
};

class OptionParser
{
public:
    enum OptionType {
        OPT_STRING = 0,
        OPT_INT32,
        OPT_UINT32,
        OPT_INT64,
        OPT_UINT64,
        OPT_BOOL,
        OPT_HELP,
    };
    enum OptionAction {
        STORE,
        STORE_TRUE,
        STORE_FALSE,
    };
public:
    typedef std::map<std::string, std::string> StrOptMap;
    typedef std::map<std::string, bool> BoolOptMap;
    typedef std::map<std::string, int32_t> IntOptMap;

public:
    OptionParser(const std::string &usageDescription = "");
    ~OptionParser();

    void addOption(const std::string &shortOpt, const std::string &longOpt, 
                   const std::string& optName, 
                   const OptionType type = OPT_STRING, bool isRequired = false);

    void addOption(const std::string &shortOpt, const std::string &longOpt, 
                   const std::string& optName, 
                   const OptionAction &action,
                   const OptionType type = OPT_BOOL, bool isRequired = false);

    void addOption(const std::string &shortOpt, const std::string &longOpt, 
                   const std::string& optName, 
                   const char* defaultValue); 

    void addOption(const std::string &shortOpt, const std::string &longOpt, 
                   const std::string& optName, 
                   const std::string& defaultValue); 

    void addOption(const std::string &shortOpt, const std::string &longOpt, 
                   const std::string& optName, 
                   const uint32_t defaultValue);

    void addOption(const std::string &shortOpt, const std::string &longOpt, 
                   const std::string& optName, 
                   const int32_t defaultValue);

    void addOption(const std::string &shortOpt, const std::string &longOpt, 
                   const std::string& optName, 
                   const bool defaultValue);


    bool parseArgs(int argc, char **argv);
    bool parseArgs(const std::string &commandString);
    bool getOptionValue(const std::string &optName, std::string &value) const;
    bool getOptionValue(const std::string &optName, bool &value) const;
    bool getOptionValue(const std::string &optName, int32_t &value) const;
    bool getOptionValue(const std::string &optName, uint32_t &value) const;
    StrOptMap getOptionValues() const;

private:
    class OptionInfo{
    public:
        OptionInfo(const std::string &optionName, OptionType type, 
                   OptionAction action, bool required,
                   const std::string &defaultValue = "") 
            : optionType(type), optionAction(action), 
              optionName(optionName), isRequired(required)
        {
            isSet = false;
        }
        ~OptionInfo() {}
        bool hasValue() {return optionAction == STORE;}
    public:
        OptionType optionType;
        OptionAction optionAction;
        std::string optionName;
        bool isRequired;
        bool isSet;
    };

private:
    void addOptionInfo(const OptionInfo &optionInfo, 
                       const std::string& shortOpt, const std::string& longOpt);
private:
    StrOptMap _strOptMap;
    IntOptMap _intOptMap;
    BoolOptMap _boolOptMap;
    std::string _usageDescription;
private:
    typedef std::map<std::string, size_t> ShortOpt2InfoMap;
    typedef std::map<std::string, size_t> LongOpt2InfoMap;
    typedef std::vector<OptionInfo> OptionInfos;
    ShortOpt2InfoMap _shortOpt2InfoMap;
    LongOpt2InfoMap _longOpt2InfoMap;
    OptionInfos _optionInfos;
private:
    void init();
    bool isValidArgs();
private:
    friend class OptionParserTest;
};

} //ps

#endif  // PS_COMMON_OPTION_PARSER_H
