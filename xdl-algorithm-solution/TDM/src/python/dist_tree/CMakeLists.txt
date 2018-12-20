add_subdirectory(cc)

find_program(PYTHON "python")
if (PYTHON)
    set(SETUP_PY    "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")
    set(DEPS        dist_tree_api)
    set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/timestamp")

    add_custom_command(OUTPUT ${OUTPUT}
                       COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cc/dist_tree_api.py "${CMAKE_CURRENT_SOURCE_DIR}/dist_tree"
                       COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_CURRENT_SOURCE_DIR} ${PYTHON} ${SETUP_PY} sdist -d ${CMAKE_CURRENT_BINARY_DIR}
                       COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                       DEPENDS ${DEPS})
    add_custom_target(dist_tree_api_target ALL DEPENDS ${OUTPUT})

    install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_CURRENT_SOURCE_DIR} ${PYTHON} ${SETUP_PY} install)")
endif()
