# Finds Google Protocol Buffers library and compilers and extends
# the standard cmake script with version and python generation support

set(PROTOBUF_GENERATE_CPP_APPEND_PATH TRUE)

################################################################################################
# Modification of standard 'protobuf_generate_cpp()' with output dir parameter and python support
# Usage:
#   protobuf_generate_cpp_py(<output_dir> <srcs_var> <hdrs_var> <python_var> <proto_files>)
function(protobuf_generate_cpp_py output_dir srcs_var hdrs_var python_var work_path proto_path)
  if(NOT ARGN)
    message(SEND_ERROR "Error: protobuf_generate_cpp_py() called without any proto files")
    return()
  endif()

  if(PROTOBUF_GENERATE_CPP_APPEND_PATH)
    # Create an include path for each file specified
    foreach(fil ${ARGN})
      get_filename_component(abs_fil ${fil} ABSOLUTE)
      get_filename_component(abs_path ${abs_fil} PATH)
      list(FIND _protoc_include ${abs_path} _contains_already)
      if(${_contains_already} EQUAL -1)
        list(APPEND _protoc_include -I ${abs_path})
      endif()
    endforeach()
  else()
    set(_protoc_include -I ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  if(DEFINED PROTOBUF_IMPORT_DIRS)
    foreach(dir ${PROTOBUF_IMPORT_DIRS})
      get_filename_component(abs_path ${dir} ABSOLUTE)
      list(FIND _protoc_include ${abs_path} _contains_already)
      if(${_contains_already} EQUAL -1)
        list(APPEND _protoc_include -I ${abs_path})
      endif()
    endforeach()
  endif()

  set(${srcs_var})
  set(${hdrs_var})
  set(${python_var})
  foreach(fil ${ARGN})
    get_filename_component(abs_fil ${fil} ABSOLUTE)
    get_filename_component(fil_we ${fil} NAME_WE)
	string(REPLACE ${work_path}/ "" o_fil ${abs_fil})
	string(REPLACE "${fil_we}.proto" "" o_fil_path ${o_fil})

    list(APPEND ${srcs_var} "${o_fil_path}/${fil_we}.pb.cc")
    list(APPEND ${hdrs_var} "${o_fil_path}/${fil_we}.pb.h")
    list(APPEND ${python_var} "${o_fil_path}/${fil_we}_pb2.py")
	
    add_custom_command(
      OUTPUT "${o_fil_path}/${fil_we}.pb.cc"
             "${o_fil_path}/${fil_we}.pb.h"
             "${o_fil_path}/${fil_we}_pb2.py"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${output_dir}"
      COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --cpp_out    ${output_dir} ${o_fil} --proto_path ${proto_path} ${_protoc_include}
      COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --python_out ${output_dir} ${o_fil} --proto_path ${proto_path} ${_protoc_include}
      DEPENDS ${abs_fil}
      WORKING_DIRECTORY ${work_path}
      COMMENT "Running C++/Python protocol buffer compiler on ${o_fil}" VERBATIM )
    
    string(REPLACE "/" "_" new_file ${fil})
    add_custom_target(GenPb${new_file} ALL DEPENDS "${o_fil_path}/${fil_we}.pb.cc"
                                                   "${o_fil_path}/${fil_we}.pb.h"
                                                   "${o_fil_path}/${fil_we}_pb2.py")
    add_custom_command(
        TARGET GenPb${new_file}
        PRE_BUILD
        COMMAND
        echo "Compile protobuf"
        COMMENT "This command will be executed befor building target GenPb")
  endforeach()

  set_source_files_properties(${${srcs_var}} ${${hdrs_var}} ${${python_var}} PROPERTIES GENERATED TRUE)
  set(${srcs_var} ${${srcs_var}} PARENT_SCOPE)
  set(${hdrs_var} ${${hdrs_var}} PARENT_SCOPE)
  set(${python_var} ${${python_var}} PARENT_SCOPE)
endfunction()
