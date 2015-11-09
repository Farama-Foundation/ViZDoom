#include <boost/interprocess/ipc/message_queue.hpp>

#include "vizia_main.h"
#include "vizia_input.h"
#include "vizia_game.h"
#include "vizia_screen.h"
#include "vizia_shared_memory.h"

#include "d_main.h"
#include "d_net.h"
#include "g_game.h"
#include "doomdef.h"
#include "doomstat.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

using namespace boost::interprocess;

void Vizia_Init(){
    printf("VIZIA INIT\n");

    Vizia_SMInit();
    Vizia_ScreenInit();
    Vizia_GameVarsInit();
    Vizia_InputInit();
}

void Vizia_Close(){
    printf("VIZIA CLOSE\n");

    Vizia_ScreenClose();
    Vizia_GameVarsClose();
    Vizia_InputClose();
    Vizia_SMClose();
}

void Vizia_Tic(){
    if (gamestate == GS_LEVEL && !paused) {
        //printf("VIZIA GAME TIC\n");

        Vizia_InputTic();
        Vizia_UpdateGameVars();
        //Vizia_ScreenUpdate();
    }
}

