cmake_minimum_required(VERSION 3.12)

	set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/cmake_modules")

if( COMMAND cmake_policy )
	cmake_policy( SET CMP0003 NEW )
endif( COMMAND cmake_policy )

include( CheckCXXSourceCompiles )
include( CheckFunctionExists )
include( CheckCXXCompilerFlag )
include( CheckLibraryExists )
include( FindPkgConfig )

option( DYN_FLUIDSYNTH "Dynamically load fluidsynth" ON )

if( CMAKE_SIZEOF_VOID_P MATCHES "8" )
	set( X64 64 )
endif( CMAKE_SIZEOF_VOID_P MATCHES "8" )

# You can either use "make install" on the FMOD distribution to put it
# in standard system locations, or you can unpack the FMOD distribution
# in the root of the zdoom tree. e.g.:
# zdoom
#   docs
#   fmodapi<version>linux[64] -or simply- fmod
#   jpeg-6b
#   ...
# The recommended method is to put it in the zdoom tree, since its
# headers are unversioned. Especially now that we can't work properly
# with anything newer than 4.26.xx, you probably don't want to use
# a system-wide version.

# Construct version numbers for searching for the FMOD library on Linux.
set( MINOR_VERSIONS "50" "49" "48" "47" "46" "45" "44" "43" "42" "41"
	"40" "39" "38" "37" "36" "35" "34" "33" "32" "31" "30" "29" "28"
	"27" "26" "25" "24" "23" "22" "21" "20" "21" "19" "18" "17" "16"
	"15" "14" "13" "12" "11" "10" "09" "08" "07" "06" "05" "04" "03"
	"02" "01" "00" )
set( MAJOR_VERSIONS "44" "34" "28" "26" "24" "22" "20" )

if( NOT FMOD_DIR_VERSIONS )
	set( FMOD_DIR_VERSIONS "" )
endif( NOT FMOD_DIR_VERSIONS )
if( NOT FMOD_VERSIONS )
	set( FMOD_VERSIONS "" )
endif( NOT FMOD_VERSIONS )
if( NOT FMOD_LOCAL_INC_DIRS )
	set( FMOD_LOCAL_INC_DIRS "" )
endif( NOT FMOD_LOCAL_INC_DIRS )
if( NOT FMOD_LOCAL_LIB_DIRS )
	set( FMOD_LOCAL_LIB_DIRS "" )
endif( NOT FMOD_LOCAL_LIB_DIRS )

set( FMOD_DIR_VERSIONS ${FMOD_DIR_VERSIONS} "../fmod" )
foreach( majver ${MAJOR_VERSIONS} )
	foreach( minver ${MINOR_VERSIONS} )
		set( FMOD_VERSIONS ${FMOD_VERSIONS} "fmodex${X64}-4.${majver}.${minver}" )
		set( FMOD_DIR_VERSIONS ${FMOD_DIR_VERSIONS} "${CMAKE_HOME_DIRECTORY}/fmodapi4${majver}${minver}linux${X64}" )
	endforeach( minver ${MINOR_VERSIONS} )
	foreach( dir ${FMOD_DIR_VERSIONS} )
		set( FMOD_LOCAL_INC_DIRS ${FMOD_LOCAL_INC_DIRS} "${dir}/api/inc" )
		set( FMOD_LOCAL_LIB_DIRS ${FMOD_LOCAL_LIB_DIRS} "${dir}/api/lib" )
	endforeach( dir ${FMOD_DIR_VERSIONS} )
endforeach( majver ${MAJOR_VERSIONS} )

if( NOT ZDOOM_LIBS )
	set( ZDOOM_LIBS "" )
endif( NOT ZDOOM_LIBS )

if( WIN32 )
	if( X64 )
		set( WIN_TYPE Win64 )
		set( XBITS x64 )
	else( X64 )
		set( WIN_TYPE Win32 )
		set( XBITS x86 )
	endif( X64 )

	add_definitions( -D_WIN32 )

	set( FMOD_SEARCH_PATHS
		"C:/Program Files/FMOD SoundSystem/FMOD Programmers API ${WIN_TYPE}/api"
		"C:/Program Files (x86)/FMOD SoundSystem/FMOD Programmers API ${WIN_TYPE}/api"
		)
	set( FMOD_INC_PATH_SUFFIXES PATH_SUFFIXES inc )
	set( FMOD_LIB_PATH_SUFFIXES PATH_SUFFIXES lib )

	message( "Could not find xinput.h. XInput will be disabled." )

	find_path( D3D_INCLUDE_DIR d3d9.h
		PATHS ENV DXSDK_DIR
		PATH_SUFFIXES Include )
	if( NOT D3D_INCLUDE_DIR )
		message( SEND_ERROR "Could not find DirectX 9 header files" )
	else( NOT D3D_INCLUDE_DIR )
		include_directories( ${D3D_INCLUDE_DIR} )
	endif( NOT D3D_INCLUDE_DIR )

	find_path( XINPUT_INCLUDE_DIR xinput.h
		PATHS ENV DXSDK_DIR
		PATH_SUFFIXES Include )
	if( NOT XINPUT_INCLUDE_DIR )
		message( WARNING "Could not find xinput.h. XInput will be disabled." )
		add_definitions( -DNO_XINPUT )
	else( NOT XINPUT_INCLUDE_DIR )
		include_directories( ${XINPUT_INCLUDE_DIR} )
	endif( NOT XINPUT_INCLUDE_DIR )

	find_library( DX_dxguid_LIBRARY dxguid
		PATHS ENV DXSDK_DIR
		PATH_SUFFIXES Lib Lib/${XBITS} )
	find_library( DX_dinput8_LIBRARY dinput8
		PATHS ENV DXSDK_DIR
		PATH_SUFFIXES Lib Lib/${XBITS} )

	set( DX_LIBS_FOUND YES )
	if( NOT DX_dxguid_LIBRARY )
		set( DX_LIBS_FOUND NO )
	endif( NOT DX_dxguid_LIBRARY )
	if( NOT DX_dinput8_LIBRARY )
		set( DX_LIBS_FOUND NO )
	endif( NOT DX_dinput8_LIBRARY )

	if( NOT DX_LIBS_FOUND )
		message( FATAL_ERROR "Could not find DirectX 9 libraries" )
	endif( NOT DX_LIBS_FOUND )

	set( ZDOOM_LIBS
		wsock32
		winmm
		"${DX_dxguid_LIBRARY}"
		"${DX_dinput8_LIBRARY}"
		ole32
		user32
		gdi32
		comctl32
		comdlg32
		ws2_32
		setupapi
		oleaut32
		DelayImp )
else( WIN32 )
	if( APPLE )
		set( FMOD_SEARCH_PATHS "/Developer/FMOD Programmers API Mac/api" )
		set( FMOD_INC_PATH_SUFFIXES PATH_SUFFIXES inc )
		set( FMOD_LIB_PATH_SUFFIXES PATH_SUFFIXES lib )
		set( NO_GTK ON )
	else( APPLE )
		option( NO_GTK "Disable GTK+ dialogs (Not applicable to Windows)" )
		option( VALGRIND "Add special Valgrind sequences to self-modifying code" )

		set( FMOD_SEARCH_PATHS
			/usr/local/include
			/usr/local/include/fmodex
			/usr/include
			/usr/include/fmodex
			/opt/local/include
			/opt/local/include/fmodex
			/opt/include
			/opt/include/fmodex )
		set( FMOD_INC_PATH_SUFFIXES PATH_SUFFIXES fmodex )

		# Use GTK+ for the IWAD picker, if available.
		if( NOT NO_GTK )
			pkg_check_modules( GTK2 gtk+-2.0 )
			if( GTK2_FOUND )
				set( ZDOOM_LIBS ${ZDOOM_LIBS} ${GTK2_LIBRARIES} )
				include_directories( ${GTK2_INCLUDE_DIRS} )
				link_directories( ${GTK2_LIBRARY_DIRS} )
			else( GTK2_FOUND )
				set( NO_GTK ON )
			endif( GTK2_FOUND )
		endif( NOT NO_GTK )
	endif( APPLE )
	
	if( NO_GTK )
		add_definitions( -DNO_GTK=1 )
	endif( NO_GTK )

	# Non-Windows version needs SDL2
	find_package( SDL2 REQUIRED )
	include_directories( "${SDL2_INCLUDE_DIR}" )
	set( ZDOOM_LIBS ${ZDOOM_LIBS} "${SDL2_LIBRARY}" )
endif( WIN32 )


if( NOT NO_OPENAL )
    find_package( OpenAL REQUIRED )
	mark_as_advanced(CLEAR OPENAL_INCLUDE_DIR)
    if( OPENAL_INCLUDE_DIR )
		message( STATUS "OpenAL include files found at ${OPENAL_INCLUDE_DIR}" )
        include_directories( ${OPENAL_INCLUDE_DIR} )
	else ( OPENAL_INCLUDE_DIR )
		message( STATUS "Could not find OpenAL include files" )
    endif( OPENAL_INCLUDE_DIR )
endif( NOT NO_OPENAL )

if( NOT NO_FMOD )
	# Search for FMOD include files
	if( NOT WIN32 )
		find_path( FMOD_INCLUDE_DIR fmod.hpp
			PATHS ${FMOD_LOCAL_INC_DIRS} )
	endif( NOT WIN32 )

	if( NOT FMOD_INCLUDE_DIR )
		find_path( FMOD_INCLUDE_DIR fmod.hpp
			PATHS ${FMOD_SEARCH_PATHS}
			${FMOD_INC_PATH_SUFFIXES} )
	endif( NOT FMOD_INCLUDE_DIR )

	if( FMOD_INCLUDE_DIR )
		message( STATUS "FMOD include files found at ${FMOD_INCLUDE_DIR}" )
		include_directories( "${FMOD_INCLUDE_DIR}" )
	else( FMOD_INCLUDE_DIR )
		message( STATUS "Could not find FMOD include files" )
		set( NO_FMOD ON )
	endif( FMOD_INCLUDE_DIR )
endif( NOT NO_FMOD )

if( NOT NO_FMOD )
	# Decide on the name of the FMOD library we want to use.
	if( NOT FMOD_LIB_NAME AND MSVC )
		set( FMOD_LIB_NAME fmodex${X64}_vc )
	endif( NOT FMOD_LIB_NAME AND MSVC )

	if( NOT FMOD_LIB_NAME AND BORLAND )
		set( FMOD_LIB_NAME fmodex${X64}_bc )
	endif( NOT FMOD_LIB_NAME AND BORLAND )

	if( NOT FMOD_LIB_NAME )
		set( FMOD_LIB_NAME fmodex${X64} )
	endif( NOT FMOD_LIB_NAME )

	# Search for FMOD library
	if( WIN32 OR APPLE )
		find_library( FMOD_LIBRARY ${FMOD_LIB_NAME}
			PATHS ${FMOD_SEARCH_PATHS}
			${FMOD_LIB_PATH_SUFFIXES} )
	else( WIN32 OR APPLE )
		find_library( FMOD_LIBRARY
			NAMES ${FMOD_VERSIONS}
			PATHS ${FMOD_LOCAL_LIB_DIRS} )
	endif( WIN32 OR APPLE )

	if( FMOD_LIBRARY )
		message( STATUS "FMOD library found at ${FMOD_LIBRARY}" )
		set( ZDOOM_LIBS ${ZDOOM_LIBS} "${FMOD_LIBRARY}" )
	else( FMOD_LIBRARY )
		message( STATUS "Could not find FMOD library" )
		set( NO_FMOD ON )
	endif( FMOD_LIBRARY )
endif( NOT NO_FMOD )

if( NO_FMOD )
	add_definitions( -DNO_FMOD=1 )
endif( NO_FMOD )
if( NO_OPENAL )
	add_definitions( -DNO_OPENAL=1 )

	set(MPG123_FOUND NO)
	set(SNDFILE_FOUND NO)
else( NO_OPENAL )
	# Search for libSndFile
	find_package( SndFile )

	# Search for libmpg123
	find_package( MPG123 )
endif( NO_OPENAL )

# Search for FluidSynth

find_package( FluidSynth )

# Set up flags for GCC

if( ZD_CMAKE_COMPILER_IS_GNUCXX_COMPATIBLE )
	if( PROFILE )
		set( CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -pg" )
		set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pg" )
		set( CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -pg" )
		set( CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -pg" )
	endif( PROFILE )

	#Disabled For Boost
	#set( REL_CXX_FLAGS "-fno-rtti" )
	if( NOT PROFILE AND NOT APPLE )
		# On OS X frame pointers are required for exception handling, at least with Clang
		set( REL_CXX_FLAGS "${REL_CXX_FLAGS} -fomit-frame-pointer" )
	endif( NOT PROFILE AND NOT APPLE )
	set( CMAKE_CXX_FLAGS_RELEASE "${REL_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}" )
	set( CMAKE_CXX_FLAGS_MINSIZEREL "${REL_CXX_FLAGS} ${CMAKE_CXX_FLAGS_MINSIZEREL}" )
	set( CMAKE_CXX_FLAGS_RELWITHDEBINFO "${REL_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}" )

	if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.5")
                set( CMAKE_C_FLAGS "-Wno-unused-result ${CMAKE_C_FLAGS}" )
		set( CMAKE_CXX_FLAGS "-Wno-unused-result ${CMAKE_CXX_FLAGS}" )
        endif(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.5")
	set( CMAKE_C_FLAGS "-Wall -Wextra -Wno-unused -Wno-unused-parameter -Wno-missing-field-initializers ${CMAKE_C_FLAGS}" )
	set( CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused -Wno-unused-parameter -Wno-missing-field-initializers ${CMAKE_CXX_FLAGS}" )

	# Remove extra warnings when using the official DirectX headers.
	# Also, TDM-GCC 4.4.0 no longer accepts glibc-style printf formats as valid,
	# which is a royal pain. The previous version I had been using was fine with them.
	if( WIN32 )
		set( CMAKE_CXX_FLAGS "-Wno-unknown-pragmas -Wno-comment -Wno-format ${CMAKE_CXX_FLAGS}" )
	endif( WIN32 )

	if( NOT NO_STRIP )
		set (CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -s" )
		set (CMAKE_EXE_LINKER_FLAGS_MINSIZEREL "${CMAKE_EXE_LINKER_FLAGS_MINSIZEREL} -s" )
	endif( NOT NO_STRIP )
endif( ZD_CMAKE_COMPILER_IS_GNUCXX_COMPATIBLE )

# Check for functions that may or may not exist.

CHECK_FUNCTION_EXISTS( filelength FILELENGTH_EXISTS )
if( FILELENGTH_EXISTS )
	add_definitions( -DHAVE_FILELENGTH=1 )
endif( FILELENGTH_EXISTS )

CHECK_FUNCTION_EXISTS( strupr STRUPR_EXISTS )
if( NOT STRUPR_EXISTS )
	add_definitions( -DNEED_STRUPR=1 )
endif( NOT STRUPR_EXISTS )

CHECK_FUNCTION_EXISTS( stricmp STRICMP_EXISTS )
if( NOT STRICMP_EXISTS )
	add_definitions( -Dstricmp=strcasecmp )
endif( NOT STRICMP_EXISTS )

CHECK_FUNCTION_EXISTS( strnicmp STRNICMP_EXISTS )
if( NOT STRNICMP_EXISTS )
	add_definitions( -Dstrnicmp=strncasecmp )
endif( NOT STRNICMP_EXISTS )

if( NOT MSVC )
	add_definitions( -D__forceinline=inline )
endif( NOT MSVC )

# Fix stat in v140_xp (broken in RTM and Update 1 so far)
if( MSVC AND MSVC_VERSION EQUAL 1900 AND CMAKE_GENERATOR_TOOLSET STREQUAL "v140_xp" )
	add_definitions( -D_stat64i32=VS14Stat )
endif( MSVC AND MSVC_VERSION EQUAL 1900 AND CMAKE_GENERATOR_TOOLSET STREQUAL "v140_xp" )

if( UNIX )
	CHECK_LIBRARY_EXISTS( rt clock_gettime "" CLOCK_GETTIME_IN_RT )
	if( NOT CLOCK_GETTIME_IN_RT )
		CHECK_FUNCTION_EXISTS( clock_gettime CLOCK_GETTIME_EXISTS )
		if( NOT CLOCK_GETTIME_EXISTS )
			message( STATUS "Could not find clock_gettime. Timing statistics will not be available." )
			add_definitions( -DNO_CLOCK_GETTIME )
		endif( NOT CLOCK_GETTIME_EXISTS )
	else( NOT CLOCK_GETTIME_IN_RT )
		set( ZDOOM_LIBS ${ZDOOM_LIBS} rt )
	endif( NOT CLOCK_GETTIME_IN_RT )
endif( UNIX )

CHECK_CXX_SOURCE_COMPILES(
	"#include <stdarg.h>
	int main() { va_list list1, list2; va_copy(list1, list2); return 0; }"
	HAS_VA_COPY )
if( NOT HAS_VA_COPY )
	CHECK_CXX_SOURCE_COMPILES(
		"#include <stdarg.h>
		int main() { va_list list1, list2; __va_copy(list1, list2); return 0; }"
		HAS___VA_COPY )
	if( HAS___VA_COPY )
		add_definitions( -Dva_copy=__va_copy )
	else( HAS___VA_COPY )
		add_definitions( -DNO_VA_COPY )
	endif( HAS___VA_COPY )
endif( NOT HAS_VA_COPY )

# Flags

if( BACKPATCH )
	add_definitions( -DBACKPATCH )
endif( BACKPATCH )

# Update gitinfo.h

#add_custom_target( revision_check ALL
#	COMMAND updaterevision src/gitinfo.h
#	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
#	DEPENDS updaterevision )

# Libraries ZDoom needs

find_package(Boost COMPONENTS thread date_time chrono REQUIRED)

# Libraries ZDoom needs

message( STATUS "Fluid synth libs: ${FLUIDSYNTH_LIBRARIES}" )
set( ZDOOM_LIBS ${ZDOOM_LIBS} "${ZLIB_LIBRARIES}" "${JPEG_LIBRARIES}" "${BZIP2_LIBRARIES}" "${GME_LIBRARIES}"  ${Boost_LIBRARIES} )
include_directories( "${ZLIB_INCLUDE_DIR}" "${BZIP2_INCLUDE_DIR}" "${LZMA_INCLUDE_DIR}" "${JPEG_INCLUDE_DIR}" "${GME_INCLUDE_DIR}" ${Boost_INCLUDE_DIR} )

if( SNDFILE_FOUND )
    set( ZDOOM_LIBS ${ZDOOM_LIBS} "${SNDFILE_LIBRARIES}" )
    include_directories( "${SNDFILE_INCLUDE_DIRS}" )
endif( SNDFILE_FOUND )
if( MPG123_FOUND )
    set( ZDOOM_LIBS ${ZDOOM_LIBS} "${MPG123_LIBRARIES}" )
    include_directories( "${MPG123_INCLUDE_DIR}" )
endif( MPG123_FOUND )
if( NOT DYN_FLUIDSYNTH)
	if( FLUIDSYNTH_FOUND )
		set( ZDOOM_LIBS ${ZDOOM_LIBS} "${FLUIDSYNTH_LIBRARIES}" )
		include_directories( "${FLUIDSYNTH_INCLUDE_DIR}" )
	endif( FLUIDSYNTH_FOUND )
else( NOT DYN_FLUIDSYNTH )
	set( ZDOOM_LIBS ${ZDOOM_LIBS} ${CMAKE_DL_LIBS} )
endif( NOT DYN_FLUIDSYNTH )

# Start defining source files for ZDoom
set( PLAT_WIN32_SOURCES
	win32/eaxedit.cpp
	win32/fb_d3d9.cpp
	win32/fb_d3d9_wipe.cpp
	win32/fb_ddraw.cpp
	win32/hardware.cpp
	win32/helperthread.cpp
	win32/i_cd.cpp
	win32/i_crash.cpp
	win32/i_input.cpp
	win32/i_keyboard.cpp
	win32/i_mouse.cpp
	win32/i_dijoy.cpp
	win32/i_rawps2.cpp
	win32/i_xinput.cpp
	win32/i_main.cpp
	win32/i_movie.cpp
	win32/i_system.cpp
	win32/st_start.cpp
	win32/win32video.cpp )
set( PLAT_POSIX_SOURCES
	posix/i_cd.cpp
	posix/i_movie.cpp
	posix/i_steam.cpp )
set( PLAT_SDL_SOURCES
	posix/sdl/crashcatcher.c
	posix/sdl/hardware.cpp
	posix/sdl/i_gui.cpp
	posix/sdl/i_input.cpp
	posix/sdl/i_joystick.cpp
	posix/sdl/i_main.cpp
	posix/sdl/i_system.cpp
	posix/sdl/i_timer.cpp
	posix/sdl/sdlvideo.cpp
	posix/sdl/st_start.cpp )
set( PLAT_OSX_SOURCES
	posix/osx/iwadpicker_cocoa.mm
	posix/osx/zdoom.icns )

if( WIN32 )
	set( SYSTEM_SOURCES_DIR win32 )
	set( SYSTEM_SOURCES ${PLAT_WIN32_SOURCES} )
	set( OTHER_SYSTEM_SOURCES ${PLAT_POSIX_SOURCES} ${PLAT_SDL_SOURCES} ${PLAT_OSX_SOURCES} ${PLAT_COCOA_SOURCES} )

	if( ZD_CMAKE_COMPILER_IS_GNUCXX_COMPATIBLE )
		# CMake is not set up to compile and link rc files with GCC. :(
		add_custom_command( OUTPUT zdoom-rc.o
			COMMAND windres -o zdoom-rc.o -i ${CMAKE_CURRENT_SOURCE_DIR}/win32/zdoom.rc
			DEPENDS win32/zdoom.rc )
		set( SYSTEM_SOURCES ${SYSTEM_SOURCES} zdoom-rc.o )
	else( ZD_CMAKE_COMPILER_IS_GNUCXX_COMPATIBLE )
		set( SYSTEM_SOURCES ${SYSTEM_SOURCES} win32/zdoom.rc )
	endif( ZD_CMAKE_COMPILER_IS_GNUCXX_COMPATIBLE )
elseif( APPLE )
	set( SYSTEM_SOURCES_DIR posix posix/sdl )
	set( SYSTEM_SOURCES ${PLAT_SDL_SOURCES} )
	set( PLAT_OSX_SOURCES ${PLAT_OSX_SOURCES} posix/sdl/i_system.mm )
	set( OTHER_SYSTEM_SOURCES ${PLAT_WIN32_SOURCES} ${PLAT_COCOA_SOURCES} )

	set( SYSTEM_SOURCES ${SYSTEM_SOURCES} ${PLAT_POSIX_SOURCES} ${PLAT_OSX_SOURCES} "${FMOD_LIBRARY}" )

	set_source_files_properties( posix/osx/zdoom.icns PROPERTIES MACOSX_PACKAGE_LOCATION Resources )
	set_source_files_properties( "${FMOD_LIBRARY}" PROPERTIES MACOSX_PACKAGE_LOCATION Frameworks )
else( WIN32 )
	set( SYSTEM_SOURCES_DIR posix posix/sdl )
	set( SYSTEM_SOURCES ${PLAT_POSIX_SOURCES} ${PLAT_SDL_SOURCES} )
	set( OTHER_SYSTEM_SOURCES ${PLAT_WIN32_SOURCES} ${PLAT_OSX_SOURCES} ${PLAT_COCOA_SOURCES} )
endif( WIN32 )

add_custom_command( OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/xlat_parser.c ${CMAKE_CURRENT_BINARY_DIR}/xlat_parser.h
	COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/xlat/xlat_parser.y .
	COMMAND lemon xlat_parser.y
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	DEPENDS lemon ${CMAKE_CURRENT_SOURCE_DIR}/xlat/xlat_parser.y )

add_custom_command( OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/sc_man_scanner.h
	COMMAND re2c --no-generation-date -s -o ${CMAKE_CURRENT_BINARY_DIR}/sc_man_scanner.h ${CMAKE_CURRENT_SOURCE_DIR}/sc_man_scanner.re
	DEPENDS re2c ${CMAKE_CURRENT_SOURCE_DIR}/sc_man_scanner.re )

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

if( SNDFILE_FOUND )
    add_definitions( -DHAVE_SNDFILE )
endif( SNDFILE_FOUND )
if( MPG123_FOUND )
    add_definitions( -DHAVE_MPG123 )
endif( MPG123_FOUND )
if( DYN_FLUIDSYNTH )
	add_definitions( -DHAVE_FLUIDSYNTH -DDYN_FLUIDSYNTH )
elseif( FLUIDSYNTH_FOUND )
	add_definitions( -DHAVE_FLUIDSYNTH )
endif( DYN_FLUIDSYNTH )

# Project files should be aware of the header files. We can GLOB these since
# there's generally a new cpp for every header so this file will get changed
if( WIN32 )
	set( EXTRA_HEADER_DIRS win32/*.h )
elseif( APPLE )
	set( EXTRA_HEADER_DIRS posix/*.h posix/sdl/*.h )
else( WIN32 )
	set( EXTRA_HEADER_DIRS posix/*.h posix/sdl/*.h )
endif( WIN32 )
file( GLOB HEADER_FILES
	${EXTRA_HEADER_DIRS}
	fragglescript/*.h
	g_doom/*.h
	g_heretic/*.h
	g_hexen/*.h
	g_raven/*.h
	g_shared/*.h
	g_strife/*.h
	intermission/*.h
	menu/*.h
	oplsynth/*.h
	oplsynth/dosbox/*.h
	posix/*.h
	posix/sdl/*.h
	r_data/*.h
	resourcefiles/*.h
	sfmt/*.h
	sound/*.h
	textures/*.h
	thingdef/*.h
	xlat/*.h
	*.h
)

# These files will be flagged as "headers" so that they appear in project files
# without being compiled.
set( NOT_COMPILED_SOURCE_FILES
	${OTHER_SYSTEM_SOURCES}
	sc_man_scanner.h
	sc_man_scanner.re
	g_doom/a_arachnotron.cpp
	g_doom/a_archvile.cpp
	g_doom/a_bossbrain.cpp
	g_doom/a_bruiser.cpp
	g_doom/a_cacodemon.cpp
	g_doom/a_cyberdemon.cpp
	g_doom/a_demon.cpp
	g_doom/a_doomimp.cpp
	g_doom/a_doomweaps.cpp
	g_doom/a_fatso.cpp
	g_doom/a_keen.cpp
	g_doom/a_lostsoul.cpp
	g_doom/a_painelemental.cpp
	g_doom/a_possessed.cpp
	g_doom/a_revenant.cpp
	g_doom/a_scriptedmarine.cpp
	g_doom/a_spidermaster.cpp
	g_heretic/a_chicken.cpp
	g_heretic/a_dsparil.cpp
	g_heretic/a_hereticartifacts.cpp
	g_heretic/a_hereticimp.cpp
	g_heretic/a_hereticweaps.cpp
	g_heretic/a_ironlich.cpp
	g_heretic/a_knight.cpp
	g_heretic/a_wizard.cpp
	g_hexen/a_bats.cpp
	g_hexen/a_bishop.cpp
	g_hexen/a_blastradius.cpp
	g_hexen/a_boostarmor.cpp
	g_hexen/a_centaur.cpp
	g_hexen/a_clericflame.cpp
	g_hexen/a_clericholy.cpp
	g_hexen/a_clericmace.cpp
	g_hexen/a_clericstaff.cpp
	g_hexen/a_dragon.cpp
	g_hexen/a_fighteraxe.cpp
	g_hexen/a_fighterhammer.cpp
	g_hexen/a_fighterplayer.cpp
	g_hexen/a_fighterquietus.cpp
	g_hexen/a_firedemon.cpp
	g_hexen/a_flechette.cpp
	g_hexen/a_flies.cpp
	g_hexen/a_fog.cpp
	g_hexen/a_healingradius.cpp
	g_hexen/a_heresiarch.cpp
	g_hexen/a_hexenspecialdecs.cpp
	g_hexen/a_iceguy.cpp
	g_hexen/a_korax.cpp
	g_hexen/a_magecone.cpp
	g_hexen/a_magelightning.cpp
	g_hexen/a_magestaff.cpp
	g_hexen/a_pig.cpp
	g_hexen/a_serpent.cpp
	g_hexen/a_spike.cpp
	g_hexen/a_summon.cpp
	g_hexen/a_teleportother.cpp
	g_hexen/a_wraith.cpp
	g_strife/a_acolyte.cpp
	g_strife/a_alienspectres.cpp
	g_strife/a_coin.cpp
	g_strife/a_crusader.cpp
	g_strife/a_entityboss.cpp
	g_strife/a_inquisitor.cpp
	g_strife/a_oracle.cpp
	g_strife/a_programmer.cpp
	g_strife/a_reaver.cpp
	g_strife/a_rebels.cpp
	g_strife/a_sentinel.cpp
	g_strife/a_stalker.cpp
	g_strife/a_strifeitems.cpp
	g_strife/a_strifeweapons.cpp
	g_strife/a_templar.cpp
	g_strife/a_thingstoblowup.cpp
	g_shared/sbarinfo_commands.cpp
	xlat/xlat_parser.y
	xlat_parser.c
	xlat_parser.h
)

add_executable( vizdoom WIN32 MACOSX_BUNDLE
		${HEADER_FILES}
		${NOT_COMPILED_SOURCE_FILES}
		__autostart.cpp
		${SYSTEM_SOURCES}
		${X86_SOURCES}
		x86.cpp
		actorptrselect.cpp
		am_map.cpp
		b_bot.cpp
		b_func.cpp
		b_game.cpp
		b_move.cpp
		b_think.cpp
		bbannouncer.cpp
		c_bind.cpp
		c_cmds.cpp
		c_console.cpp
		c_consolebuffer.cpp
		c_cvars.cpp
		c_dispatch.cpp
		c_expr.cpp
		cmdlib.cpp
		colormatcher.cpp
		compatibility.cpp
		configfile.cpp
		ct_chat.cpp
		d_dehacked.cpp
		d_iwad.cpp
		d_main.cpp
		d_net.cpp
		d_netinfo.cpp
		d_protocol.cpp
		decallib.cpp
		dobject.cpp
		dobjgc.cpp
		dobjtype.cpp
		doomdef.cpp
		doomstat.cpp
		dsectoreffect.cpp
		dthinker.cpp
		f_wipe.cpp
		farchive.cpp
		files.cpp
		g_doomedmap.cpp
		g_game.cpp
		g_hub.cpp
		g_level.cpp
		g_mapinfo.cpp
		g_skill.cpp
		gameconfigfile.cpp
		gi.cpp
		gitinfo.cpp
		hu_scores.cpp
		i_net.cpp
		info.cpp
		keysections.cpp
		lumpconfigfile.cpp
		m_alloc.cpp
		m_argv.cpp
		m_bbox.cpp
		m_cheat.cpp
		m_joy.cpp
		m_misc.cpp
		m_png.cpp
		m_random.cpp
		m_specialpaths.cpp
		memarena.cpp
		md5.cpp
		name.cpp
		nodebuild.cpp
		nodebuild_classify_nosse2.cpp
		nodebuild_events.cpp
		nodebuild_extract.cpp
		nodebuild_gl.cpp
		nodebuild_utility.cpp
		pathexpander.cpp
		p_3dfloors.cpp
		p_3dmidtex.cpp
		p_acs.cpp
		p_buildmap.cpp
		p_ceiling.cpp
		p_conversation.cpp
		p_doors.cpp
		p_effect.cpp
		p_enemy.cpp
		p_floor.cpp
		p_glnodes.cpp
		p_interaction.cpp
		p_lights.cpp
		p_linkedsectors.cpp
		p_lnspec.cpp
		p_map.cpp
		p_maputl.cpp
		p_mobj.cpp
		p_pillar.cpp
		p_plats.cpp
		p_pspr.cpp
		p_saveg.cpp
		p_sectors.cpp
		p_setup.cpp
		p_sight.cpp
		p_slopes.cpp
		p_spec.cpp
		p_states.cpp
		p_switch.cpp
		p_tags.cpp
		p_teleport.cpp
		p_terrain.cpp
		p_things.cpp
		p_tick.cpp
		p_trace.cpp
		p_udmf.cpp
		p_usdf.cpp
		p_user.cpp
		p_writemap.cpp
		p_xlat.cpp
		parsecontext.cpp
		po_man.cpp
		r_swrenderer.cpp
		r_utility.cpp
		r_3dfloors.cpp
		r_bsp.cpp
		r_draw.cpp
		r_drawt.cpp
		r_main.cpp
		r_plane.cpp
		r_segs.cpp
		r_sky.cpp
		r_things.cpp
		s_advsound.cpp
		s_environment.cpp
		s_playlist.cpp
		s_sndseq.cpp
		s_sound.cpp
		sc_man.cpp
		st_stuff.cpp
		statistics.cpp
		stats.cpp
		stringtable.cpp
		strnatcmp.c
		tables.cpp
		teaminfo.cpp
		tempfiles.cpp
		v_blend.cpp
		v_collection.cpp
		v_draw.cpp
		v_font.cpp
		v_palette.cpp
		v_pfx.cpp
		v_text.cpp
		v_video.cpp
		w_wad.cpp
		wi_stuff.cpp
		zstrformat.cpp
		zstring.cpp
		GuillotineBinPack.cpp
		SkylineBinPack.cpp
		g_doom/a_doommisc.cpp
		g_heretic/a_hereticmisc.cpp
		g_hexen/a_hexenmisc.cpp
		g_raven/a_artitele.cpp
		g_raven/a_minotaur.cpp
		g_strife/a_strifestuff.cpp
		g_strife/strife_sbar.cpp
		g_shared/a_action.cpp
		g_shared/a_armor.cpp
		g_shared/a_artifacts.cpp
		g_shared/a_bridge.cpp
		g_shared/a_camera.cpp
		g_shared/a_debris.cpp
		g_shared/a_decals.cpp
		g_shared/a_fastprojectile.cpp
		g_shared/a_flashfader.cpp
		g_shared/a_fountain.cpp
		g_shared/a_hatetarget.cpp
		g_shared/a_keys.cpp
		g_shared/a_lightning.cpp
		g_shared/a_mapmarker.cpp
		g_shared/a_morph.cpp
		g_shared/a_movingcamera.cpp
		g_shared/a_pickups.cpp
		g_shared/a_puzzleitems.cpp
		g_shared/a_quake.cpp
		g_shared/a_randomspawner.cpp
		g_shared/a_secrettrigger.cpp
		g_shared/a_sectoraction.cpp
		g_shared/a_setcolor.cpp
		g_shared/a_skies.cpp
		g_shared/a_soundenvironment.cpp
		g_shared/a_soundsequence.cpp
		g_shared/a_spark.cpp
		g_shared/a_specialspot.cpp
		g_shared/a_waterzone.cpp
		g_shared/a_weaponpiece.cpp
		g_shared/a_weapons.cpp
		g_shared/hudmessages.cpp
		g_shared/sbarinfo.cpp
		g_shared/sbar_mugshot.cpp
		g_shared/shared_hud.cpp
		g_shared/shared_sbar.cpp
		intermission/intermission.cpp
		intermission/intermission_parse.cpp
		menu/colorpickermenu.cpp
		menu/joystickmenu.cpp
		menu/listmenu.cpp
		menu/loadsavemenu.cpp
		menu/menu.cpp
		menu/menudef.cpp
		menu/menuinput.cpp
		menu/messagebox.cpp
		menu/optionmenu.cpp
		menu/playerdisplay.cpp
		menu/playermenu.cpp
		menu/readthis.cpp
		menu/videomenu.cpp
		oplsynth/fmopl.cpp
		oplsynth/mlopl.cpp
		oplsynth/mlopl_io.cpp
		oplsynth/music_opldumper_mididevice.cpp
		oplsynth/music_opl_mididevice.cpp
		oplsynth/opl_mus_player.cpp
		oplsynth/dosbox/opl.cpp
		oplsynth/OPL3.cpp
		oplsynth/nukedopl3.cpp
		resourcefiles/ancientzip.cpp
		resourcefiles/file_7z.cpp
		resourcefiles/file_grp.cpp
		resourcefiles/file_lump.cpp
		resourcefiles/file_rff.cpp
		resourcefiles/file_wad.cpp
		resourcefiles/file_zip.cpp
		resourcefiles/file_pak.cpp
		resourcefiles/file_directory.cpp
		resourcefiles/resourcefile.cpp
		sfmt/SFMT.cpp
		sound/fmodsound.cpp
		sound/i_music.cpp
		sound/i_sound.cpp
		sound/mpg123_decoder.cpp
		sound/music_cd.cpp
		sound/music_dumb.cpp
		sound/music_gme.cpp
		sound/music_mus_midiout.cpp
		sound/music_smf_midiout.cpp
		sound/music_hmi_midiout.cpp
		sound/music_xmi_midiout.cpp
		sound/music_midistream.cpp
		sound/music_midi_base.cpp
		sound/music_midi_timidity.cpp
		sound/music_mus_opl.cpp
		sound/music_stream.cpp
		sound/music_fluidsynth_mididevice.cpp
		sound/music_softsynth_mididevice.cpp
		sound/music_timidity_mididevice.cpp
		sound/music_wildmidi_mididevice.cpp
		sound/music_win_mididevice.cpp
		sound/oalsound.cpp
		sound/sndfile_decoder.cpp
		sound/music_pseudo_mididevice.cpp
		textures/animations.cpp
		textures/anim_switches.cpp
		textures/automaptexture.cpp
		textures/bitmap.cpp
		textures/buildtexture.cpp
		textures/canvastexture.cpp
		textures/ddstexture.cpp
		textures/flattexture.cpp
		textures/imgztexture.cpp
		textures/jpegtexture.cpp
		textures/multipatchtexture.cpp
		textures/patchtexture.cpp
		textures/pcxtexture.cpp
		textures/pngtexture.cpp
		textures/rawpagetexture.cpp
		textures/emptytexture.cpp
		textures/texture.cpp
		textures/texturemanager.cpp
		textures/tgatexture.cpp
		textures/warptexture.cpp
		thingdef/olddecorations.cpp
		thingdef/thingdef.cpp
		thingdef/thingdef_codeptr.cpp
		thingdef/thingdef_data.cpp
		thingdef/thingdef_exp.cpp
		thingdef/thingdef_expression.cpp
		thingdef/thingdef_function.cpp
		thingdef/thingdef_parse.cpp
		thingdef/thingdef_properties.cpp
		thingdef/thingdef_states.cpp
		timidity/common.cpp
		timidity/instrum.cpp
		timidity/instrum_dls.cpp
		timidity/instrum_font.cpp
		timidity/instrum_sf2.cpp
		timidity/mix.cpp
		timidity/playmidi.cpp
		timidity/resample.cpp
		timidity/timidity.cpp
		wildmidi/file_io.cpp
		wildmidi/gus_pat.cpp
		wildmidi/reverb.cpp
		wildmidi/wildmidi_lib.cpp
		wildmidi/wm_error.cpp
		xlat/parse_xlat.cpp
		fragglescript/t_fspic.cpp
		fragglescript/t_func.cpp
		fragglescript/t_load.cpp
		fragglescript/t_oper.cpp
		fragglescript/t_parse.cpp
		fragglescript/t_prepro.cpp
		fragglescript/t_script.cpp
		fragglescript/t_spec.cpp
		fragglescript/t_variable.cpp
		fragglescript/t_cmd.cpp
		r_data/colormaps.cpp
		r_data/sprites.cpp
		r_data/voxels.cpp
		r_data/renderstyle.cpp
		r_data/r_interpolate.cpp
		r_data/r_translate.cpp
		zzautozend.cpp
		viz_depth.cpp
		viz_game.cpp
		viz_input.cpp
		viz_labels.cpp
		viz_main.cpp
		viz_message_queue.cpp
		viz_buffers.cpp
		viz_shared_memory.cpp
		viz_system.cpp)

set_source_files_properties( xlat/parse_xlat.cpp PROPERTIES OBJECT_DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/xlat_parser.c" )
set_source_files_properties( sc_man.cpp PROPERTIES OBJECT_DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/sc_man_scanner.h" )
set_source_files_properties( ${NOT_COMPILED_SOURCE_FILES} PROPERTIES HEADER_FILE_ONLY TRUE )

if(${CMAKE_SYSTEM_NAME} STREQUAL "SunOS")
	# [BL] Solaris requires these to be explicitly linked.
	set( ZDOOM_LIBS ${ZDOOM_LIBS} nsl socket)
endif(${CMAKE_SYSTEM_NAME} STREQUAL "SunOS")

target_link_libraries( vizdoom ${ZDOOM_LIBS} gdtoa dumb lzma )
include_directories( .
	g_doom
	g_heretic
	g_hexen
	g_raven
	g_strife
	g_shared
	oplsynth
	sound
	textures
	thingdef
	timidity
	xlat
	../gdtoa
	../dumb/include
	${CMAKE_BINARY_DIR}/gdtoa
	${SYSTEM_SOURCES_DIR} )

#add_dependencies( vizdoom revision_check )

set_target_properties( vizdoom PROPERTIES
	RUNTIME_OUTPUT_NAME ${ZDOOM_EXE_NAME}
	RUNTIME_OUTPUT_DIRECTORY ${ZDOOM_OUTPUT_DIR}
	RUNTIME_OUTPUT_NAME_RELEASE ${ZDOOM_EXE_NAME}
	RUNTIME_OUTPUT_DIRECTORY_RELEASE ${ZDOOM_OUTPUT_DIR}
	RUNTIME_OUTPUT_NAME_DEBUG ${ZDOOM_EXE_NAME}d
	RUNTIME_OUTPUT_DIRECTORY_DEBUG ${ZDOOM_OUTPUT_DIR}
	RUNTIME_OUTPUT_NAME_MINSIZEREL ${ZDOOM_EXE_NAME}msr
	RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${ZDOOM_OUTPUT_DIR}
	RUNTIME_OUTPUT_NAME_RELWITHDEBINFO ${ZDOOM_EXE_NAME}rd
	RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${ZDOOM_OUTPUT_DIR}
)

if( MSVC )
	option( ZDOOM_GENERATE_MAPFILE "Generate .map file for debugging." OFF )
	if( ZDOOM_GENERATE_MAPFILE )
		set_target_properties(vizdoom PROPERTIES LINK_FLAGS "/MANIFEST:NO /DELAYLOAD:\"fmodex${X64}.dll\" /DELAYLOAD:\"openal32.dll\" /DELAYLOAD:\"libmpg123-0.dll\" /DELAYLOAD:\"libsndfile-1.dll\" /MAP")
	else( ZDOOM_GENERATE_MAPFILE )
		set_target_properties(vizdoom PROPERTIES LINK_FLAGS "/MANIFEST:NO /DELAYLOAD:\"fmodex${X64}.dll\" /DELAYLOAD:\"openal32.dll\" /DELAYLOAD:\"libmpg123-0.dll\" /DELAYLOAD:\"libsndfile-1.dll\"")
	endif( ZDOOM_GENERATE_MAPFILE )

	if( NO_GENERATOR_EXPRESSIONS )
		add_custom_command(TARGET vizdoom POST_BUILD
			COMMAND "mt.exe" -manifest \"${CMAKE_CURRENT_SOURCE_DIR}\\win32\\zdoom.exe.manifest\" -outputresource:\"$(TargetDir)$(TargetFileName)\"\;\#1
			COMMENT "Adding manifest..."
		)
	else( NO_GENERATOR_EXPRESSIONS )
		add_custom_command(TARGET vizdoom POST_BUILD
			COMMAND "mt.exe" -manifest \"${CMAKE_CURRENT_SOURCE_DIR}\\win32\\zdoom.exe.manifest\" -outputresource:\"$<TARGET_FILE:vizdoom>\"\;\#1
			COMMENT "Adding manifest..."
		)
	endif( NO_GENERATOR_EXPRESSIONS )

	create_default_target_launcher( vizdoom WORKING_DIRECTORY ${ZDOOM_OUTPUT_DIR} )
endif( MSVC )

if( NOT WIN32 )
	FILE( WRITE ${CMAKE_CURRENT_BINARY_DIR}/link-make "if [ ! -e ${ZDOOM_OUTPUT_DIR}/${ZDOOM_EXE_NAME} ]; then ln -sf ${CMAKE_CURRENT_BINARY_DIR}/${ZDOOM_EXE_NAME} ${ZDOOM_OUTPUT_DIR}/${ZDOOM_EXE_NAME}; fi" )
	add_custom_command( TARGET vizdoom POST_BUILD
		COMMAND chmod +x ${CMAKE_CURRENT_BINARY_DIR}/link-make
		COMMAND /bin/sh -c ${CMAKE_CURRENT_BINARY_DIR}/link-make )
endif( NOT WIN32 )
if( CMAKE_COMPILER_IS_GNUCXX )
	# GCC misoptimizes this file
	set_source_files_properties( oplsynth/fmopl.cpp PROPERTIES COMPILE_FLAGS "-fno-tree-dominator-opts -fno-tree-fre" )
endif( CMAKE_COMPILER_IS_GNUCXX )

if( APPLE )
	set_target_properties(vizdoom PROPERTIES
		LINK_FLAGS "-framework Carbon -framework Cocoa -framework IOKit -framework OpenGL"
		MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/posix/osx/zdoom-info.plist" )

	if( NOT NO_FMOD )
		# Fix fmod link so that it can be found in the app bundle.
		find_program( OTOOL otool HINTS "/usr/bin" "${OSX_DEVELOPER_ROOT}/usr/bin" )
		find_program( INSTALL_NAME_TOOL install_name_tool HINTS "/usr/bin" "${OSX_DEVELOPER_ROOT}/usr/bin" )
		execute_process( COMMAND "${OTOOL}" -L "${FMOD_LIBRARY}"
			COMMAND grep "libfmodex.dylib (compat"
			COMMAND head -n1
			COMMAND awk "{print $1}"
			OUTPUT_VARIABLE FMOD_LINK
			OUTPUT_STRIP_TRAILING_WHITESPACE )
		add_custom_command( TARGET vizdoom POST_BUILD
			COMMAND "${INSTALL_NAME_TOOL}" -change "${FMOD_LINK}" @executable_path/../Frameworks/libfmodex.dylib "$<TARGET_FILE:vizdoom>"
			COMMENT "Relinking FMOD Ex" )
	endif( NOT NO_FMOD )
endif( APPLE )

source_group("Audio Files" REGULAR_EXPRESSION "^${CMAKE_CURRENT_SOURCE_DIR}/sound/.+")
source_group("Audio Files\\OPL Synth" REGULAR_EXPRESSION "^${CMAKE_CURRENT_SOURCE_DIR}/oplsynth/.+")
source_group("Audio Files\\OPL Synth\\DOSBox" FILES oplsynth/dosbox/opl.cpp oplsynth/dosbox/opl.h)
source_group("Audio Files\\Timidity\\Headers" REGULAR_EXPRESSION "^${CMAKE_CURRENT_SOURCE_DIR}/timidity/.+\\.h$")
source_group("Audio Files\\Timidity\\Source" REGULAR_EXPRESSION "^${CMAKE_CURRENT_SOURCE_DIR}/timidity/.+\\.cpp$")
source_group("Audio Files\\WildMidi\\Headers" REGULAR_EXPRESSION "^${CMAKE_CURRENT_SOURCE_DIR}/wildmidi/.+\\.h$")
source_group("Audio Files\\WildMidi\\Source" REGULAR_EXPRESSION "^${CMAKE_CURRENT_SOURCE_DIR}/wildmidi/.+\\.cpp$")
source_group("Decorate++" REGULAR_EXPRESSION "^${CMAKE_CURRENT_SOURCE_DIR}/thingdef/.+")
source_group("FraggleScript" REGULAR_EXPRESSION "^${CMAKE_CURRENT_SOURCE_DIR}/fragglescript/.+")
source_group("Games\\Doom Game" REGULAR_EXPRESSION "^${CMAKE_CURRENT_SOURCE_DIR}/g_doom/.+")
source_group("Games\\Heretic Game" REGULAR_EXPRESSION "^${CMAKE_CURRENT_SOURCE_DIR}/g_heretic/.+")
source_group("Games\\Hexen Game" REGULAR_EXPRESSION "^${CMAKE_CURRENT_SOURCE_DIR}/g_hexen/.+")
source_group("Games\\Raven Shared" REGULAR_EXPRESSION "^${CMAKE_CURRENT_SOURCE_DIR}/g_raven/.+")
source_group("Games\\Strife Game" REGULAR_EXPRESSION "^${CMAKE_CURRENT_SOURCE_DIR}/g_strife/.+")
source_group("Intermission" REGULAR_EXPRESSION "^${CMAKE_CURRENT_SOURCE_DIR}/intermission/.+")
source_group("Menu" REGULAR_EXPRESSION "^${CMAKE_CURRENT_SOURCE_DIR}/menu/.+")
source_group("Render Core\\Render Headers" REGULAR_EXPRESSION "^${CMAKE_CURRENT_SOURCE_DIR}/r_.+\\.h$")
source_group("Render Core\\Render Sources" REGULAR_EXPRESSION "^${CMAKE_CURRENT_SOURCE_DIR}/r_.+\\.cpp$")
source_group("Render Data\\Resource Headers" REGULAR_EXPRESSION "^${CMAKE_CURRENT_SOURCE_DIR}/r_data/.+\\.h$")
source_group("Render Data\\Resource Sources" REGULAR_EXPRESSION "^${CMAKE_CURRENT_SOURCE_DIR}/r_data/.+\\.cpp$")
source_group("Render Data\\Textures" REGULAR_EXPRESSION "^${CMAKE_CURRENT_SOURCE_DIR}/textures/.+")
source_group("Render Interface" FILES r_defs.h r_renderer.h r_sky.cpp r_sky.h r_state.h r_utility.cpp r_utility.h)
source_group("Resource Files" REGULAR_EXPRESSION "^${CMAKE_CURRENT_SOURCE_DIR}/resourcefiles/.+")
source_group("POSIX Files" REGULAR_EXPRESSION "^${CMAKE_CURRENT_SOURCE_DIR}/posix/.+")
source_group("OS X Files" REGULAR_EXPRESSION "^${CMAKE_CURRENT_SOURCE_DIR}/posix/osx/.+")
source_group("SDL Files" REGULAR_EXPRESSION "^${CMAKE_CURRENT_SOURCE_DIR}/posix/sdl/.+")
source_group("SFMT" REGULAR_EXPRESSION "^${CMAKE_CURRENT_SOURCE_DIR}/sfmt/.+")
source_group("Shared Game" REGULAR_EXPRESSION "^${CMAKE_CURRENT_SOURCE_DIR}/g_shared/.+")
source_group("Versioning" FILES version.h win32/zdoom.rc)
source_group("Win32 Files" REGULAR_EXPRESSION "^${CMAKE_CURRENT_SOURCE_DIR}/win32/.+")
source_group("Xlat" REGULAR_EXPRESSION "^${CMAKE_CURRENT_SOURCE_DIR}/xlat/.+" FILES ${CMAKE_CURRENT_BINARY_DIR}/xlat_parser.c ${CMAKE_CURRENT_BINARY_DIR}/xlat_parser.h)
source_group("Source Files" FILES ${CMAKE_CURRENT_BINARY_DIR}/sc_man_scanner.h sc_man_scanner.re)
