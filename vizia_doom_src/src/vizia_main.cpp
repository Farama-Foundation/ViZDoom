#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/thread.hpp>
#include <boost/thread/thread.hpp>

#include "vizia_main.h"
#include "vizia_defines.h"
#include "vizia_input.h"
#include "vizia_game.h"
#include "vizia_screen.h"
#include "vizia_shared_memory.h"
#include "vizia_message_queue.h"

#include "d_main.h"
#include "d_net.h"
#include "g_game.h"
#include "doomdef.h"
#include "doomstat.h"
#include "doomtype.h"
#include "c_console.h"

#include "d_player.h"
#include "d_event.h"

namespace b = boost;
namespace bt = boost::this_thread;

CVAR (Bool, vizia_controlled, false, CVAR_NOSET)
CVAR (Bool, vizia_async, false, CVAR_NOSET)
CVAR (String, vizia_instance_id, "0", CVAR_NOSET)
CVAR (Int, vizia_screen_format, 0, CVAR_NOSET)
CVAR (Bool, vizia_no_console, false, CVAR_NOSET)
CVAR (Bool, vizia_window_hidden, false, CVAR_NOSET)
CVAR (Bool, vizia_no_x_server, false, CVAR_NOSET)
CVAR (Bool, vizia_allow_input, false, CVAR_NOSET)

int vizia_time = 0;
bool viziaNextTic = false;
bool viziaUpdate = false;
unsigned int viziaLastUpdate = 0;

void Vizia_Init(){
    Printf("Vizia_Init: Instance id: %s\n", *vizia_instance_id);

    if(*vizia_controlled) {
        Printf("Vizia_Init: Init message queues\n");
        Vizia_MQInit(*vizia_instance_id);

        Printf("Vizia_Init: Init shared memory\n");
        Vizia_SMInit(*vizia_instance_id);

        Vizia_InputInit();
        Vizia_GameVarsInit();

        Vizia_ScreenInit();

        viziaNextTic = true;
        viziaUpdate = true;
    }
}

void Vizia_Close(){
    if(*vizia_controlled) {
        Vizia_InputClose();
        Vizia_GameVarsClose();
        Vizia_ScreenClose();

        Vizia_SMClose();
        Vizia_MQClose();
    }
}

void Vizia_AsyncStartTic(){
    try{
        bt::interruption_point();
    }
    catch(b::thread_interrupted &ex ){
        exit(0);
    }

    if (*vizia_controlled && !Vizia_IsPaused()){
        Vizia_MQTic();
        if(viziaNextTic) Vizia_InputTic();
    }
}

void Vizia_Tic(){

    try{
        bt::interruption_point();
    }
    catch(b::thread_interrupted &ex ){
        exit(0);
    }

    if (*vizia_controlled && !Vizia_IsPaused()){

        if(viziaUpdate) {
            Vizia_Update();
        }
        if(viziaNextTic) {
            Vizia_GameVarsTic();
            Vizia_MQSend(VIZIA_MSG_CODE_DOOM_DONE);
            viziaNextTic = false;
        }

        if(!*vizia_async){
            Vizia_MQTic();
            Vizia_InputTic();
        }
    }
}

void Vizia_Update(){
    D_Display();
    Vizia_ScreenUpdate();
    viziaLastUpdate = VIZIA_TIME;
    viziaUpdate = false;
}

bool Vizia_IsPaused(){
    return menuactive != MENU_Off;

    //&& (gamestate == GS_LEVEL || gamestate == GS_TITLELEVEL || gamestate == GS_INTERMISSION || gamestate == GS_FINALE)
    //&& !paused && !pauseext && menuactive == MENU_Off && ConsoleState != c_down && ConsoleState != c_falling
}

