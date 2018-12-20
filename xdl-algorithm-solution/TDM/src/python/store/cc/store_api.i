%module store_api

%include "cdata.i"
%include "typemaps.i"
%include "cmalloc.i"

%include "stl.i"
%template(_string_list) std::vector< std::string >;

%{
#include "include/api.h"
%}

#define API(name) STORE_API_##name

typedef void* store_handler;

store_handler API(new)(const std::string& config);

void API(load)(store_handler handler, const std::string& filename);

void API(close)(store_handler handler);

int API(put)(store_handler handler,
             const std::string& key,
             const std::string& value);

int API(get)(store_handler handler,
             const std::string& key, std::string* value);

//////////////  Batch operation interface  ///////////////

int API(mget)(store_handler handler,
              const std::vector<std::string>& keys,
              std::vector<std::string>* values);

int API(mput)(store_handler handler,
              const std::vector<std::string>& keys,
              const std::vector<std::string>& values);

%inline %{
  //////////// helper functions /////////////
  std::string* new_string() {
    return new std::string();
  }

  std::string string_value(const std::string* value) {
    return *value;
  }

  void free_string(std::string* str) {
    delete str;
  }

  std::vector<std::string>* new_string_vector() {
    return new std::vector<std::string>();
  }

  std::vector<std::string>
  string_vector_value(const std::vector<std::string>* values) {
    return *values;
  }

  void free_string_vector(std::vector<std::string>* values) {
    delete values;
  }
%}
