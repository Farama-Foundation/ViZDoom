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

#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/thread.hpp>

#include "vizdoom_main.h"
#include "vizdoom_defines.h"
#include "vizdoom_input.h"
#include "vizdoom_game.h"
#include "vizdoom_screen.h"
#include "vizdoom_shared_memory.h"
#include "vizdoom_message_queue.h"

#include "d_main.h"
#include "d_net.h"

namespace b = boost;
namespace bt = boost::this_thread;

CVAR (Bool, vizdoom_controlled, false, CVAR_NOSET)
CVAR (Bool, vizdoom_async, false, CVAR_NOSET)
CVAR (String, vizdoom_instance_id, "0", CVAR_NOSET)
CVAR (Int, vizdoom_screen_format, 0, CVAR_NOSET)
CVAR (Bool, vizdoom_no_console, false, CVAR_NOSET)
CVAR (Bool, vizdoom_no_sound, false, CVAR_NOSET)
CVAR (Bool, vizdoom_window_hidden, false, CVAR_NOSET)
CVAR (Bool, vizdoom_no_x_server, false, CVAR_NOSET)
CVAR (Bool, vizdoom_allow_input, false, CVAR_NOSET)
CVAR (Bool, vizdoom_nocheat, false, CVAR_NOSET)

int vizdoom_time = 0;
bool vizdoomNextTic = false;
bool vizdoomUpdate = false;
unsigned int vizdoomLastUpdate = 0;

void ViZDoom_Init(){
    Printf("ViZDoom_Init: Instance id: %s\n", *vizdoom_instance_id);

    if(*vizdoom_controlled) {
        Printf("ViZDoom_Init: Init message queues\n");
        ViZDoom_MQInit(*vizdoom_instance_id);

        Printf("ViZDoom_Init: Init shared memory\n");
        ViZDoom_SMInit(*vizdoom_instance_id);

        ViZDoom_InputInit();
        ViZDoom_GameVarsInit();

        ViZDoom_ScreenInit();

        vizdoomNextTic = true;
        vizdoomUpdate = true;
    }
}

void ViZDoom_Close(){
    if(*vizdoom_controlled) {
        ViZDoom_InputClose();
        ViZDoom_GameVarsClose();
        ViZDoom_ScreenClose();

        ViZDoom_SMClose();
        ViZDoom_MQClose();
    }
}

void ViZDoom_AsyncStartTic(){
    try{
        bt::interruption_point();
    }
    catch(b::thread_interrupted &ex ){
        exit(0);
    }

    if (*vizdoom_controlled && !ViZDoom_IsPaused()){
        ViZDoom_MQTic();
        if(vizdoomNextTic) ViZDoom_InputTic();
    }
}

void ViZDoom_Tic(){

    try{
        bt::interruption_point();
    }
    catch(b::thread_interrupted &ex ){
        exit(0);
    }

    if (*vizdoom_controlled && !ViZDoom_IsPaused()){

        NetUpdate();

        if(vizdoomUpdate) {
            ViZDoom_Update();
        }
        if(vizdoomNextTic) {
            ViZDoom_GameVarsTic();
            ViZDoom_MQSend(VIZDOOM_MSG_CODE_DOOM_DONE);
            vizdoomNextTic = false;
        }

        if(!*vizdoom_async){
            ViZDoom_MQTic();
            ViZDoom_InputTic();
        }
    }
}

void ViZDoom_Update(){
    D_Display();
    ViZDoom_ScreenUpdate();
    vizdoomLastUpdate = VIZDOOM_TIME;
    vizdoomUpdate = false;
}

bool ViZDoom_IsPaused(){
    return menuactive != MENU_Off;

    //&& (gamestate == GS_LEVEL || gamestate == GS_TITLELEVEL || gamestate == GS_INTERMISSION || gamestate == GS_FINALE)
    //&& !paused && !pauseext && menuactive == MENU_Off && ConsoleState != c_down && ConsoleState != c_falling
}

