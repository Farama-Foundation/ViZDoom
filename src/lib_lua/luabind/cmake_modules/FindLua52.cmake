# Locate Lua library
# This module defines
#  LUA52_FOUND, if false, do not try to link to Lua
#  LUA_LIBRARIES
#  LUA_INCLUDE_DIR, where to find lua.h
#  LUA_VERSION_STRING, the version of Lua found (since CMake 2.8.8)
#
# Note that the expected include convention is
#  #include "lua.h"
# and not
#  #include <lua/lua.h>
# This is because, the lua location is not standardized and may exist
# in locations other than lua/

#=============================================================================
# CMake - Cross Platform Makefile Generator
# Copyright 2007-2009 Kitware, Inc.
#
# Adjusted for Lua 5.2 (from 5.1) by Christian Neumüller
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the names of Kitware, Inc., the Insight Software Consortium,
#   nor the names of their contributors may be used to endorse or promote
#   products derived from this software without specific prior written
#   permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#=============================================================================

FIND_PATH(LUA_INCLUDE_DIR lua.h
  HINTS
  $ENV{LUA_DIR}
  PATH_SUFFIXES include/lua52 include/lua5.2 include/lua include
  PATHS
  ~/Library/Frameworks
  /Library/Frameworks
  /sw # Fink
  /opt/local # DarwinPorts
  /opt/csw # Blastwave
  /opt
)

FIND_LIBRARY(_LUA_LIBRARY_RELEASE
  NAMES lua52 lua5.2 lua-5.2 lua
  HINTS
  $ENV{LUA_DIR}
  PATH_SUFFIXES lib64 lib
  PATHS
  ~/Library/Frameworks
  /Library/Frameworks
  /sw
  /opt/local
  /opt/csw
  /opt
)

FIND_LIBRARY(_LUA_LIBRARY_DEBUG
  NAMES lua52-d lua5.2-d lua-5.2-d lua-d
  HINTS
  $ENV{LUA_DIR}
  PATH_SUFFIXES lib64 lib
  PATHS
  ~/Library/Frameworks
  /Library/Frameworks
  /sw
  /opt/local
  /opt/csw
  /opt
)

IF(_LUA_LIBRARY_RELEASE OR _LUA_LIBRARY_DEBUG)
  IF(_LUA_LIBRARY_RELEASE AND _LUA_LIBRARY_DEBUG)
    SET(_LUA_LIBRARY optimized ${_LUA_LIBRARY_RELEASE}
                     debug     ${_LUA_LIBRARY_DEBUG})
  ELSEIF(_LUA_LIBRARY_RELEASE)
    SET(_LUA_LIBRARY ${_LUA_LIBRARY_RELEASE})
  ELSE()
    SET(_LUA_LIBRARY ${_LUA_LIBRARY_DEBUG})
  ENDIF()

  IF(UNIX AND NOT APPLE)
    FIND_LIBRARY(_LUA_MATH_LIBRARY m)
    mark_as_advanced(_LUA_MATH_LIBRARY)
  ENDIF(UNIX AND NOT APPLE)
   # For Windows and Mac, don't need to explicitly include the math library
ENDIF()

IF(_LUA_LIBRARY)
    SET(LUA_LIBRARIES
        "${_LUA_LIBRARY}" "${_LUA_MATH_LIBRARY}" CACHE STRING "Lua 5.2 Libraries")
ENDIF(_LUA_LIBRARY)


IF(LUA_INCLUDE_DIR AND EXISTS "${LUA_INCLUDE_DIR}/lua.h")
  FILE(STRINGS "${LUA_INCLUDE_DIR}/lua.h" lua_version_str REGEX "^#define[ \t]+LUA_RELEASE[ \t]+\"Lua .+\"")

  STRING(REGEX REPLACE "^#define[ \t]+LUA_RELEASE[ \t]+\"Lua ([^\"]+)\".*" "\\1" LUA_VERSION_STRING "${lua_version_str}")
  UNSET(lua_version_str)
ENDIF()

INCLUDE(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
# handle the QUIETLY and REQUIRED arguments and set LUA_FOUND to TRUE if
# all listed variables are TRUE
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Lua52
                                  REQUIRED_VARS LUA_INCLUDE_DIR LUA_LIBRARIES
                                  VERSION_VAR LUA_VERSION_STRING)

MARK_AS_ADVANCED(LUA_INCLUDE_DIR LUA_LIBRARIES LUA_LIBRARY LUA_MATH_LIBRARY
                 _LUA_LIBRARY_RELEASE _LUA_LIBRARY_DEBUG)
