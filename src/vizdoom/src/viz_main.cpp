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

#include "viz_main.h"
#include "viz_defines.h"
#include "viz_input.h"
#include "viz_game.h"
#include "viz_screen.h"
#include "viz_shared_memory.h"
#include "viz_message_queue.h"

#include "d_main.h"
#include "d_net.h"

namespace b = boost;
namespace bt = boost::this_thread;

CVAR (Bool, viz_debug, true, CVAR_NOSET)

CVAR (Bool, viz_controlled, false, CVAR_NOSET)
CVAR (String, viz_instance_id, "0", CVAR_NOSET)
CVAR (Bool, viz_async, false, CVAR_NOSET)
CVAR (Bool, viz_allow_input, false, CVAR_NOSET)

CVAR (Int, viz_screen_format, 0, CVAR_NOSET)
CVAR (Bool, viz_depth_buffer, false, CVAR_NOSET)
CVAR (Bool, viz_map_buffer, false, CVAR_NOSET)
CVAR (Bool, viz_labals, false, CVAR_NOSET)
CVAR (Bool, viz_render_all, false, CVAR_NOSET)
CVAR (Bool, viz_window_hidden, false, CVAR_NOSET)
CVAR (Bool, viz_noxserver, false, CVAR_NOSET)
CVAR (Bool, viz_noconsole, false, CVAR_NOSET)
CVAR (Bool, viz_nosound, false, CVAR_NOSET)

CVAR (Bool, viz_loop_map, false, CVAR_NOSET)
CVAR (Bool, viz_nocheat, false, CVAR_NOSET)

int vizTime = 0;
bool vizNextTic = false;
bool vizUpdate = false;
unsigned int vizLastUpdate = 0;

void VIZ_Init(){
    Printf("VIZ_Init: Instance id: %s\n", *viz_instance_id);

    if(*viz_controlled) {

        VIZ_MQInit(*viz_instance_id);
        VIZ_SMInit(*viz_instance_id);

        VIZ_GameStateInit();
        VIZ_InputInit();
        VIZ_ScreenInit();

        vizNextTic = true;
        vizUpdate = true;
    }
}

void VIZ_Close(){
    if(*viz_controlled) {
        VIZ_InputClose();
        VIZ_GameStateClose();
        VIZ_ScreenClose();

        VIZ_SMClose();
        VIZ_MQClose();
    }
}

void VIZ_AsyncStartTic(){
    try{
        bt::interruption_point();
    }
    catch(b::thread_interrupted &ex ){
        exit(0);
    }

    if (*viz_controlled && !VIZ_IsPaused()){
        VIZ_MQTic();
        if(vizNextTic) VIZ_InputTic();
    }
}

void VIZ_Tic(){

    try{
        bt::interruption_point();
    }
    catch(b::thread_interrupted &ex ){
        exit(0);
    }

    if (*viz_controlled && !VIZ_IsPaused()){

        NetUpdate();

        if(vizUpdate) {
            VIZ_Update();
        }
        if(vizNextTic) {
            VIZ_GameStateTic();
            VIZ_MQSend(VIZ_MSG_CODE_DOOM_DONE);
            vizNextTic = false;
        }

        if(!*viz_async){
            VIZ_MQTic();
            VIZ_InputTic();
        }
    }
}

void VIZ_Update(){
    D_Display();
    VIZ_ScreenUpdate();
    vizLastUpdate = VIZ_TIME;
    vizUpdate = false;
}

bool VIZ_IsPaused(){
    return menuactive != MENU_Off;

    //&& (gamestate == GS_LEVEL || gamestate == GS_TITLELEVEL || gamestate == GS_INTERMISSION || gamestate == GS_FINALE)
    //&& !paused && !pauseext && menuactive == MENU_Off && ConsoleState != c_down && ConsoleState != c_falling
}

