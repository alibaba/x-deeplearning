aux_source_directory(pybind SRC_XDL_PYTHON_PYBIND)
IF (USE_PS_PLUS)
ELSE ()
   LIST (REMOVE_ITEM SRC_XDL_PYTHON_PYBIND pybind/model_server_wrapper.cc)
ENDIF ()

add_library(xdl_python_pybind SHARED ${SRC_XDL_PYTHON_PYBIND})
target_link_libraries(xdl_python_pybind ${XDL_IO_LIB} ${XDL_CORE_LIB} libprotobuf rdkafka++ ${PYTHON_LIBRARIES})

add_custom_target(python_lib ALL
  DEPENDS xdl_core xdl_python_pybind
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/python_lib/xdl"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/python_lib/xdl/include"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/python_lib/xdl/include/third_party"
  COMMAND cp -r "${CMAKE_SOURCE_DIR}/xdl" "${CMAKE_BINARY_DIR}/python_lib/xdl/include/"
  COMMAND cp -r "${PROTOBUF_INCLUDE_DIR}/google" "${CMAKE_BINARY_DIR}/python_lib/xdl/include/"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/python_lib/xdl/bin"
  COMMAND ${CMAKE_COMMAND} -E copy "${PROTOBUF_PROTOC_PATH}" "${CMAKE_BINARY_DIR}/python_lib/xdl/bin/protoc"
  COMMAND cp -r "${CMAKE_BINARY_DIR}/xdl/proto" "${CMAKE_BINARY_DIR}/python_lib/xdl/include/xdl/"
  COMMAND cp -r "${CMAKE_SOURCE_DIR}/third_party/eigen" "${CMAKE_BINARY_DIR}/python_lib/xdl/include/third_party/"
  COMMAND cp -r "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/python_lib/xdl/"
  COMMAND cp "${CMAKE_BINARY_DIR}/xdl/core/proto/*" "${CMAKE_BINARY_DIR}/python_lib/xdl/include/xdl/core/proto"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/python_lib/xdl/python/proto"
  COMMAND cp "${CMAKE_BINARY_DIR}/xdl/proto/*.py" "${CMAKE_BINARY_DIR}/python_lib/xdl/python/proto"
  COMMAND cp "${CMAKE_BINARY_DIR}/xdl/core/proto/*.py" "${CMAKE_BINARY_DIR}/python_lib/xdl/python/proto"
  COMMAND touch "${CMAKE_BINARY_DIR}/python_lib/xdl/python/proto/__init__.py"
  COMMAND sed -i "s/#{tf_backend_import}/${TF_BACKEND_IMPORT}/g" ${CMAKE_BINARY_DIR}/python_lib/xdl/python/init.py
  COMMAND sed -i "s/#{mxnet_backend_import}/${MXNET_BACKEND_IMPORT}/g" ${CMAKE_BINARY_DIR}/python_lib/xdl/python/init.py
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py" "${CMAKE_BINARY_DIR}/python_lib/xdl/__init__.py"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/__setup__.py" "${CMAKE_BINARY_DIR}/python_lib/setup.py"
  COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:xdl_core>" "${CMAKE_BINARY_DIR}/python_lib/xdl/python/pybind/$<TARGET_FILE_NAME:xdl_core>"
  COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:xdl_python_pybind>" "${CMAKE_BINARY_DIR}/python_lib/xdl/python/pybind/$<TARGET_FILE_NAME:xdl_python_pybind>"
)

IF (USE_PS_PLUS)
  set(TAG_PS python_ps_lib)
  add_custom_target(python_ps_lib ALL
    DEPENDS python_lib ps
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/python_lib/xdl/bin"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_BINARY_DIR}/ps-plus/ps" "${CMAKE_BINARY_DIR}/python_lib/xdl/bin/ps"
  )
ELSE ()
  set(TAG_PS python_no_ps_lib)
  add_custom_target(python_no_ps_lib ALL
    DEPENDS python_lib
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_BINARY_DIR}/python_lib/xdl/python/init_local.py" "${CMAKE_BINARY_DIR}/python_lib/xdl/python/init.py"
  )
ENDIF ()


add_custom_target(install_python_lib
  DEPENDS ${TAG_PS}
  COMMAND python setup.py install
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/python_lib"
)
