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

#ifndef XDL_CORE_UTILS_STRING_UTILS_H_
#define XDL_CORE_UTILS_STRING_UTILS_H_

#include <string>
#include <vector>
#include <sstream>
#include <cstdint>
#include <iomanip>
#include <map>

namespace xdl {

class StringUtils {
 public:
  static bool strToInt8(const char* str, int8_t& value);
  static bool strToUInt8(const char* str, uint8_t& value);
  static bool strToInt16(const char* str, int16_t& value);
  static bool strToUInt16(const char* str, uint16_t& value);
  static bool strToInt32(const char* str, int32_t& value);
  static bool strToUInt32(const char* str, uint32_t& value);
  static bool strToInt64(const char* str, int64_t& value);
  static bool strToUInt64(const char* str, uint64_t& value);
  static bool strToFloat(const char *str, float &value);
  static bool strToDouble(const char *str, double &value);    
  
  static std::vector<std::string> split(
      const std::string& text, 
      const std::string &sepStr, 
      bool ignoreEmpty = true);


  template<typename T>
  static std::string ToStringPrecision(T a, int precision=6) {
    return std::to_string(a);
  }

  template<typename T>
  static std::string toString(const T &x);

  template<typename T>
  static std::string toString(
      const std::vector<T> &x, 
      const std::string &delim = " ");

  template<typename T>
  static std::string toString(
      const std::vector<std::vector<T> > &x, 
      const std::string &delim1, 
      const std::string &delim2);

  static std::string toString(const double &x, int32_t precision);    

  static bool GetValueFromMap(
      const std::map<std::string, std::string>& params, 
      const std::string& key,
      std::string* value);

 private:
  static std::stringstream* getStringStream();
  static void putStringStream(std::stringstream* ss);
  friend class StringStreamWrapper;
  class StringStreamWrapper {
   public:
    StringStreamWrapper(const std::string &str = "") 
      : _ss(StringUtils::getStringStream()) {_ss->str(str);}
    ~StringStreamWrapper() {StringUtils::putStringStream(_ss);}
    template<typename T>
    StringStreamWrapper& operator << (const T &x) {
      *_ss << x;
      return *this;
    }
    template<typename T>
    StringStreamWrapper& operator >> (T &x) {
      *_ss >> x;
      return *this;
    }
    std::string str() {return _ss->str();}
    bool eof() {return _ss->eof();}
   private:
    std::stringstream *_ss;
  };
};

template<typename T>
inline std::string StringUtils::toString(const T &x) {
  StringStreamWrapper oss;
  oss << x;
  return oss.str();    
}

template<> 
inline std::string StringUtils::toString<int8_t>(const int8_t &x) {
  char buf[8] = {0,};
  snprintf(buf, sizeof(buf), "%d", x);
  std::string res(buf);
  return res;
}

template<> 
inline std::string StringUtils::toString<uint8_t>(const uint8_t &x) {
  char buf[8] = {0,};
  snprintf(buf, sizeof(buf), "%u", x);
  std::string res(buf);
  return res;
}

template<> 
inline std::string StringUtils::toString<int16_t>(const int16_t &x) {
  char buf[16] = {0,};
  snprintf(buf, sizeof(buf), "%d", x);
  std::string res(buf);
  return res;
}

template<> 
inline std::string StringUtils::toString<uint16_t>(const uint16_t &x) {
  char buf[16] = {0,};
  snprintf(buf, sizeof(buf), "%u", x);
  std::string res(buf);
  return res;
}

template<> 
inline std::string StringUtils::toString<int32_t>(const int32_t &x) {
  char buf[32] = {0,};
  snprintf(buf, sizeof(buf), "%d", x);
  std::string res(buf);
  return res;
}

template<> 
inline std::string StringUtils::toString<uint32_t>(const uint32_t &x) {
  char buf[32] = {0,};
  snprintf(buf, sizeof(buf), "%u", x);
  std::string res(buf);
  return res;
}

template<> 
inline std::string StringUtils::toString<int64_t>(const int64_t &x) {
  char buf[64] = {0,};
  snprintf(buf, sizeof(buf), "%ld", x);
  std::string res(buf);
  return res;
}

template<> 
inline std::string StringUtils::toString<uint64_t>(const uint64_t &x) {
  char buf[64] = {0,};
  snprintf(buf, sizeof(buf), "%lu", x);
  std::string res(buf);
  return res;
}

template<> 
inline std::string StringUtils::toString<float>(const float &x) {
  StringStreamWrapper oss;
  oss << std::setprecision(6) << x;
  return oss.str();
}

template<> 
inline std::string StringUtils::toString<double>(const double &x) {
  StringStreamWrapper oss;
  oss << std::setprecision(15) << x;
  return oss.str();
}


inline std::string StringUtils::toString(const double &x, 
                                         int32_t precision) {
  StringStreamWrapper oss;
  oss << std::setprecision(precision) << x;
  return oss.str();
}

template<typename T>
inline std::string StringUtils::toString(
    const std::vector<T> &x, 
    const std::string &delim) {
  StringStreamWrapper oss; 
  for (typename std::vector<T>::const_iterator it = x.begin();
       it != x.end(); ++it)
  {
    if (it != x.begin()) oss << delim;
    oss << toString((*it));
  }
  return oss.str();
}

template<typename T>
inline std::string StringUtils::toString(
    const std::vector<std::vector<T> > &x, 
    const std::string &delim1,
    const std::string &delim2) {
  std::vector<std::string> strVec;
  for (typename std::vector<std::vector<T> >::const_iterator it = x.begin();
       it != x.end(); ++it)
  {
    strVec.push_back(toString(*it, delim1));
  }    
  return toString(strVec, delim2);
}

template<>
inline std::string StringUtils::ToStringPrecision<float>(float a, int precision) {
  if(precision < 0) {
    throw std::logic_error("Pass illegal parameter precision!");
  }
  std::string temp = "";
  long b = (long)a;
  float c = a-b;
  //if a belows zero, then change its
  //integer part and float part to
  //positive
  if(a < 0) {
    b = -b;
    c = -c;
  }
  //if( !(b <= (1<<30)) ) {
  //  throw std::logic_error("Pass illegal parameter a!");
  //}
  //change a's integer part into string
  do{
    temp += (char)(b%10+'0');
    b = b/10;
  }while(b>0);
  size_t i = temp.length();
  for(size_t j=0; j<i/2; j++){
    temp[j] = temp[j] + temp[i-j-1];
    temp[i-j-1] = temp[j]-temp[i-j-1];
    temp[j] = temp[j]-temp[i-j-1];
  }
  //memorize the pointer's position
  int pointpos = temp.length();
  //convert a's float part into string
  bool not_in_zero = (long)a != 0;
  if (-1e-7 < c && c < 1e-7) {
    if (a < 0) {
      temp.insert(0, "-");
    }
    return temp;
  }
  do{
    c = c*10;
    temp += (char)((int)c + '0');
    if ((int)c != 0) {
      not_in_zero = true;
    }
    c -= (int)c;
    if (not_in_zero) {
      precision--;
    }
    if(precision == 0 && c*10 >= 5) {//computer carrier
      int len = temp.length()-1;
      while(len>=0 && (++temp[len]) > '9') {
        temp[len] = '0';
        len--;
      }
    }
  }while(precision > 0);
  //add the pointer of a float number to string
  temp.insert(pointpos, ".");
  //add negtive sign '-'
  if(a<0) {
    temp.insert(0, "-");
  }
  while(temp[temp.size()-1] == '0') {
    temp.erase(temp.size()-1);
  }
  if (temp[temp.size()-1] == '.') {
    temp.erase(temp.size()-1);    
  }
  return temp;
}

template<>
inline std::string StringUtils::ToStringPrecision<double>(double a, int precision) {
  if(precision < 0) {
    throw std::logic_error("Pass illegal parameter precision!");
  }
  std::string temp = "";
  long b = (long)a;
  double c = a-b;
  //if a belows zero, then change its
  //integer part and float part to
  //positive
  if(a < 0) {
    b = -b;
    c = -c;
  }
  if( !(b <= (1<<30)) ) {
    throw std::logic_error("Pass illegal parameter a!");
  }
  //change a's integer part into string
  do{
    temp += (char)(b%10+'0');
    b = b/10;
  }while(b>0);
  size_t i = temp.length();
  for(size_t j=0; j<i/2; j++){
    temp[j] = temp[j] + temp[i-j-1];
    temp[i-j-1] = temp[j]-temp[i-j-1];
    temp[j] = temp[j]-temp[i-j-1];
  }
  //memorize the pointer's position
  int pointpos = temp.length();
  //convert a's float part into string
  bool not_in_zero = (long)a != 0;
  if (-1e-7 < c && c < 1e-7) {
    if (a < 0) {
      temp.insert(0, "-");
    }
    return temp;
  }
  do{
    c = c*10;
    temp += (char)((int)c + '0');
    if ((int)c != 0) {
      not_in_zero = true;
    }
    c -= (int)c;
    if (not_in_zero) {
      precision--;
    }
    if(precision == 0 && c*10 >= 5) {//computer carrier
      int len = temp.length()-1;
      while(len>=0 && (++temp[len]) > '9') {
        temp[len] = '0';
        len--;
      }
    }
  }while(precision > 0);
  //add the pointer of a float number to string
  temp.insert(pointpos, ".");
  //add negtive sign '-'
  if(a<0) {
    temp.insert(0, "-");
  }
  while(temp[temp.size()-1] == '0') {
    temp.erase(temp.size()-1);
  }
  if (temp[temp.size()-1] == '.') {
    temp.erase(temp.size()-1);    
  }
  return temp;
}

} // xdl

#endif  // XDL_CORE_UTILS_STRING_UTILS_H_
