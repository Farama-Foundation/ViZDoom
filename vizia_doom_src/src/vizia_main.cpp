#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/thread.hpp>
#include <boost/thread/thread.hpp>

#include "vizia_main.h"
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
#include "c_console.h"

namespace b = boost;
namespace bt = boost::this_thread;

/*
    CVAR (type, name, init value, flags)
    in other file EXTERN_CVAR (type, name)

    CVAR_ARCHIVE		= 1,	// set to cause it to be saved to config
    CVAR_USERINFO		= 2,	// added to userinfo  when changed
    CVAR_SERVERINFO		= 4,	// added to serverinfo when changed
    CVAR_NOSET			= 8,	// don't allow change from console at all,
                                // but can be set from the command line
    CVAR_LATCH			= 16,	// save changes until server restart
    CVAR_UNSETTABLE		= 32,	// can unset this var from console
    CVAR_DEMOSAVE		= 64,	// save the value of this cvar in a demo
    CVAR_ISDEFAULT		= 128,	// is cvar unchanged since creation?
    CVAR_AUTO			= 256,	// allocated; needs to be freed when destroyed
    CVAR_NOINITCALL		= 512,	// don't call callback at game start
    CVAR_GLOBALCONFIG	= 1024,	// cvar is saved to global config section
    CVAR_VIDEOCONFIG	= 2048, // cvar is saved to video config section (not implemented)
    CVAR_NOSAVE			= 4096, // when used with CVAR_SERVERINFO, do not save var to savegame
    CVAR_MOD			= 8192,	// cvar was defined by a mod
    CVAR_IGNORE			= 16384,// do not send cvar across the network/inaccesible from ACS (dummy mod cvar)
*/

CVAR (Bool, vizia_controlled, false, CVAR_NOSET)
CVAR (Bool, vizia_singletic, true, CVAR_NOSET)
CVAR (Bool, vizia_clear_render, true, CVAR_NOSET)
CVAR (String, vizia_instance_id, "0", CVAR_NOSET)
CVAR (Int, vizia_screen_format, 0, CVAR_NOSET)
CVAR (Bool, vizia_no_console, false, CVAR_NOSET)
CVAR (Bool, vizia_no_window_soft, false, CVAR_NOSET)

void Vizia_Init(){
    printf("Vizia_Init: Instance id: %s\n", *vizia_instance_id);

    if(*vizia_controlled) {
        printf("Vizia_Init: Init message queues\n");
        Vizia_MQInit(*vizia_instance_id);

        printf("Vizia_Init: Init shared memory\n");
        Vizia_SMInit(*vizia_instance_id);

        Vizia_InputInit();
        Vizia_GameVarsInit();
        Vizia_ScreenInit();
    }

    //Vizia_MQSend(VIZIA_MSG_CODE_DOOM_READY);
}

void Vizia_Close(){
    if(*vizia_controlled) {
        Vizia_InputClose();
        Vizia_GameVarsClose();
        Vizia_ScreenClose();

        Vizia_SMClose();

        //  Vizia_MQSend(VIZIA_MSG_CODE_DOOM_CLOSE);
        Vizia_MQClose();
    }
}

void Vizia_Tic(){

    try{
        bt::interruption_point();
    }
    catch(b::thread_interrupted &ex ){
        Vizia_Command(strdup("exit"));
    }

    if (*vizia_controlled && (gamestate == GS_LEVEL || gamestate == GS_TITLELEVEL || gamestate == GS_INTERMISSION || gamestate == GS_FINALE)
            && !paused && menuactive == MENU_Off && ConsoleState != c_down && ConsoleState != c_falling ) {

        Vizia_InputTic();
        Vizia_UpdateGameVars();
        Vizia_ScreenUpdate();

        Vizia_MQTic();
    }
}
