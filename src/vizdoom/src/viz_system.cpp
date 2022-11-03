/*
 Copyright (C) 2016 by Wojciech Jaśkowski, Michał Kempka, Grzegorz Runc, Jakub Toczek, Marek Wydmuch
 Copyright (C) 2017 - 2022 by Marek Wydmuch, Michał Kempka, Wojciech Jaśkowski, and the respective contributors

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

#include "viz_system.h"
#include "viz_defines.h"

#ifdef VIZ_OS_WIN
    #define USE_WINDOWS_DWORD
    #include <Windows.h>
#else
    #include <unistd.h>
#endif

#include <boost/chrono.hpp>
#include <boost/thread.hpp>

namespace b = boost;
namespace bt = boost::this_thread;
namespace bc = boost::chrono;


/* Sleep and interruptions handling */
/*--------------------------------------------------------------------------------------------------------------------*/

void VIZ_InterruptionPoint(){
    #ifndef VIZ_OS_WIN
        try{
            bt::interruption_point();
        }
        catch(b::thread_interrupted &ex){
            exit(0);
        }
    #endif
}

void VIZ_Sleep(unsigned int ms){
    //bt::sleep_for(bc::milliseconds(ms));
    #ifdef VIZ_OS_WIN
        Sleep(ms);
    #else
        usleep(ms);
    #endif
}
