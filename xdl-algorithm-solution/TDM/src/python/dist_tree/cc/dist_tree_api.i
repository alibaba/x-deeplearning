%module dist_tree_api

%include "cdata.i"
%include "typemaps.i"
%include "cmalloc.i"

%include "stl.i"
%template(_string_list) std::vector< std::string >;

%{
#include "include/api.h"
%}

#define API(name) DIST_TREE__API_##name
typedef void* tree_handler;

tree_handler API(new)();
void API(set_prefix)(tree_handler handler, const std::string& prefix);
void API(set_store)(tree_handler handler, void* store);
void API(set_branch)(tree_handler handler, int branch);
bool API(load)(tree_handler handler);
