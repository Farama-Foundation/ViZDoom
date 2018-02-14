# Build for LuaBind
# Ryan Pavlik <rpavlik@iastate.edu>
# http://academic.cleardefinition.com/
# Iowa State University HCI Graduate Program/VRAC

cmake_minimum_required(VERSION 2.8)
set(CMAKE_LEGACY_CYGWIN_WIN32 0) # Remove when CMake >= 2.8.4 is required
project(LuaBind)

set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "9")
set(CPACK_PACKAGE_VERSION_PATCH "1")
set(CPACK_PACKAGE_VERSION
	"${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")

if(NOT LUA_FOUND AND NOT LUA51_FOUND)
	find_package(Lua51 REQUIRED)
	set(LUA_INCLUDE_DIRS "${LUA_INCLUDE_DIR}")
endif()

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
	# We are the top-level project
	option(LUABIND_INSTALL "Install the LuaBind library and headers" ON)
	option(LUABIND_BUILD_SHARED "Build luabind as a shared library?" OFF)
	include(CTest)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
  #add_definitions("-fPIC")
endif()


set(BUILD_SHARED_LIBS ${LUABIND_BUILD_SHARED})
if(BUILD_SHARED_LIBS)
	add_definitions(-DLUABIND_DYNAMIC_LINK)
endif()

include_directories(
	"${CMAKE_CURRENT_SOURCE_DIR}"
	${LUA_INCLUDE_DIRS})

add_subdirectory(src)
