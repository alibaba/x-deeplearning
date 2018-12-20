add_subdirectory(cc)

find_program(PYTHON "python")
message(STATUS "Find Python ${PYTHON}")

if (PYTHON)
    set(SETUP_PY    "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")
    set(DEPS        cc/store_api)
    set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/timestamp")

    add_custom_command(OUTPUT ${OUTPUT}
                       COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cc/store_api.py "${CMAKE_CURRENT_SOURCE_DIR}/store"
                       COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_CURRENT_SOURCE_DIR} ${PYTHON} ${SETUP_PY} sdist -d ${CMAKE_CURRENT_BINARY_DIR}
                       COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                       DEPENDS ${DEPS})
    add_custom_target(store_api_target ALL DEPENDS ${OUTPUT})
    install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_CURRENT_SOURCE_DIR} ${PYTHON} ${SETUP_PY} install)")
endif()
