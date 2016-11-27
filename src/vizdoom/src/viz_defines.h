/*
 Copyright (C) 2016 by Wojciech Jaśkowski, Michał Kempka, Grzegorz Runc, Jakub Toczek, Marek Wydmuch

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
*/

#ifndef __VIZ_DEFINES_H__
#define __VIZ_DEFINES_H__

#include <stdlib.h>
#include <boost/current_function.hpp>
#include "doomdef.h"

#define VIZ_TIME                    ((unsigned int)level.starttime + (unsigned int)level.maptime)

#define VIZ_PLAYER                  players[consoleplayer]
#define VIZ_MAX_PLAYERS             MAXPLAYERS // 8
#define VIZ_MAX_PLAYER_NAME_LEN     MAXPLAYERNAME + 1 //(15 + 1 = 16)

#define VIZ_FUNC                    BOOST_CURRENT_FUNCTION

#define VIZ_MAX_ERROR_TEXT_LEN      128
#define VIZ_MAX_DEBUG_TEXT_LEN      1024

#ifdef __linux__
    #define VIZ_OS_LINUX
#elif _WIN32
    #define VIZ_OS_WIN
#elif __APPLE__
    #define VIZ_OS_OSX
#endif

#endif
