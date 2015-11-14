#include <boost/interprocess/ipc/message_queue.hpp>

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


void Vizia_Init(){
    printf("VIZIA INIT\n");

    Vizia_MQInit();
    //Vizia_MQRecv();

    Vizia_SMInit();

    Vizia_InputInit();
    Vizia_GameVarsInit();
    Vizia_ScreenInit();

    //Vizia_MQSend(VIZIA_MSG_CODE_DOOM_READY);
}

void Vizia_Close(){
    printf("VIZIA CLOSE\n");

    Vizia_InputClose();
    Vizia_GameVarsClose();
    Vizia_ScreenClose();

    Vizia_SMClose();

    //Vizia_MQSend(VIZIA_MSG_CODE_DOOM_CLOSE);
    Vizia_MQClose();
}

void Vizia_Tic(){
    if (gamestate == GS_LEVEL && !paused){ //menuactive == MENU_Off && ConsoleState != c_down && ConsoleState != c_falling ) {

        Vizia_InputTic();
        Vizia_UpdateGameVars();
        Vizia_ScreenUpdate();

        Vizia_MQTic();
    }
}
