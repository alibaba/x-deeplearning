# tested on OSX Yosemite and Ubuntu 14.04 LTS
# handle anaconda dependencies
cmake_minimum_required(VERSION 3.3.0)

option(ANACONDA_PYTHON_VERBOSE "Anaconda dependency info" OFF)

if(NOT CMAKE_FIND_ANACONDA_PYTHON_INCLUDED)
  set(CMAKE_FIND_ANACONDA_PYTHON_INCLUDED 1)

  # find anaconda installation
  set(_cmd conda info --root)
  execute_process(
    COMMAND ${_cmd}
    RESULT_VARIABLE _r
    OUTPUT_VARIABLE _o
    ERROR_VARIABLE _e
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
    )

  if(ANACONDA_PYTHON_VERBOSE)
    message("Executing conda info --root")
    message("_r = ${_r}")
    message("_o = ${_o}")
    message("_e = ${_e}")
  endif()

  IF(IS_DIRECTORY ${_o})
     set(ANACONDA_PYTHON_FOUND True)
  endif()

  if(ANACONDA_PYTHON_FOUND)
    set( ANACONDA_PYTHON_DIR ${_o} )
    message( "Found anaconda root directory ${ANACONDA_PYTHON_DIR}" )

    # find python version
    set(_cmd python --version)
    execute_process(
      COMMAND ${_cmd}
      RESULT_VARIABLE _r
      OUTPUT_VARIABLE _o
      ERROR_VARIABLE _o
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_STRIP_TRAILING_WHITESPACE
      )

    string (REGEX MATCH "Python ([0-9]+)[.]([0-9]+)[.]([0-9]+)" _py_version_found "${_o}")
    set( _py_version_major ${CMAKE_MATCH_1} )
    set( _py_version_minor ${CMAKE_MATCH_2} )
    set( _py_version_patch ${CMAKE_MATCH_3} )
    set( ANACONDA_PYTHON_VERSION ${_py_version_major}.${_py_version_minor} )

    if( ${_py_version_major} MATCHES 2 )
      set( _py_ext "")
    else()
      set( _py_ext "m")
    endif()

    set(_py_id "python${ANACONDA_PYTHON_VERSION}${_py_ext}")

    if( NOT DEFINED ENV{CONDA_DEFAULT_ENV} )
      set( env_CONDA_DEFAULT_ENV "root" )
      message("Could not find anaconda environment setting; using default root" )
    else()
      set( env_CONDA_DEFAULT_ENV $ENV{CONDA_DEFAULT_ENV} )
    endif()

    message( "Using anaconda ${env_CONDA_DEFAULT_ENV} environment" )
    if( env_CONDA_DEFAULT_ENV STREQUAL "root" )
      set(PYTHON_INCLUDE_DIR "${ANACONDA_PYTHON_DIR}/include/${_py_id}" CACHE INTERNAL "")
      set(PYTHON_LIBRARY "${ANACONDA_PYTHON_DIR}/lib/lib${_py_id}${CMAKE_SHARED_LIBRARY_SUFFIX}" CACHE INTERNAL "")
    else()
      set(PYTHON_INCLUDE_DIR "${ANACONDA_PYTHON_DIR}/envs/${env_CONDA_DEFAULT_ENV}/include/${_py_id}" CACHE INTERNAL "")
      set(PYTHON_LIBRARY "${ANACONDA_PYTHON_DIR}/envs/${env_CONDA_DEFAULT_ENV}/lib/lib${_py_id}${CMAKE_SHARED_LIBRARY_SUFFIX}" CACHE INTERNAL "")
    endif()

    set(PYTHON_INCLUDE_DIRS "${PYTHON_INCLUDE_DIR}")
    set(PYTHON_LIBRARIES "${PYTHON_LIBRARY}")

    set(FOUND_PYTHONLIBS TRUE)
  else()
    message( "Not found: anaconda root directory..." )
    message( "Trying system python install..." )
    find_package(PythonLibs REQUIRED)
  endif()

  message( "PYTHON_INCLUDE_DIR = ${PYTHON_INCLUDE_DIR}")
  message( "PYTHON_LIBRARY = ${PYTHON_LIBRARY}")
endif()
