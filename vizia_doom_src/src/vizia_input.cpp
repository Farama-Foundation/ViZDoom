#include "vizia_input.h"
#include "vizia_shared_memory.h"
#include "vizia_message_queue.h"

#include "d_main.h"
#include "g_game.h"
#include "d_player.h"
#include "d_event.h"
#include "c_bind.h"
#include "c_console.h"
#include "c_dispatch.h"

bip::mapped_region *viziaInputSMRegion;
ViziaInputStruct *viziaLastInput;
ViziaInputStruct *viziaInput;

void Vizia_Mouse(int x, int y){
    if(x) G_AddViewPitch (x);
    if(y) G_AddViewAngle (y);
}

bool Vizia_HasCounterBT(int button){
    switch(button){
        case VIZIA_BT_MOVE_RIGHT :
        case VIZIA_BT_MOVE_LEFT :
        case VIZIA_BT_MOVE_BACK :
        case VIZIA_BT_MOVE_FORWARD :
        case VIZIA_BT_TURN_RIGHT :
        case VIZIA_BT_TURN_LEFT :
        case VIZIA_BT_LOOK_UP :
        case VIZIA_BT_LOOK_DOWN :
        case VIZIA_BT_MOVE_UP :
        case VIZIA_BT_MOVE_DOWN :
            return true;
        default :
            return false;
    }
}

int Vizia_CounterBT(int button){
    switch(button){
        case VIZIA_BT_MOVE_RIGHT : return VIZIA_BT_MOVE_LEFT;
        case VIZIA_BT_MOVE_LEFT : return VIZIA_BT_MOVE_RIGHT;
        case VIZIA_BT_MOVE_BACK : return VIZIA_BT_MOVE_FORWARD;
        case VIZIA_BT_MOVE_FORWARD : return VIZIA_BT_MOVE_BACK;
        case VIZIA_BT_TURN_RIGHT : return VIZIA_BT_TURN_LEFT;
        case VIZIA_BT_TURN_LEFT : return VIZIA_BT_TURN_RIGHT;
        case VIZIA_BT_LOOK_UP : return VIZIA_BT_LOOK_DOWN;
        case VIZIA_BT_LOOK_DOWN : return VIZIA_BT_LOOK_UP;
        case VIZIA_BT_MOVE_UP : return VIZIA_BT_MOVE_DOWN;
        case VIZIA_BT_MOVE_DOWN : return VIZIA_BT_MOVE_UP;
        default : return -1;
    }
}

char* Vizia_BTToCommand(int button, bool state){
    switch(button){
        case VIZIA_BT_ATTACK : return Vizia_GetCommandWithState(strdup(" attack"), state);
        case VIZIA_BT_USE : return Vizia_GetCommandWithState(strdup(" use"), state);
        case VIZIA_BT_JUMP : return Vizia_GetCommandWithState(strdup(" jump"), state);
        case VIZIA_BT_CROUCH : return Vizia_GetCommandWithState(strdup(" crouch"), state);
        case VIZIA_BT_TURN180 : return strdup("turn180");
        case VIZIA_BT_ALTATTACK : return Vizia_GetCommandWithState(strdup(" altattack"), state);
        case VIZIA_BT_RELOAD : return Vizia_GetCommandWithState(strdup(" reload"), state);
        case VIZIA_BT_ZOOM : return Vizia_GetCommandWithState(strdup(" zoom"), state);

        case VIZIA_BT_SPEED : return Vizia_GetCommandWithState(strdup(" speed"), state);
        case VIZIA_BT_STRAFE : return Vizia_GetCommandWithState(strdup(" strafe"), state);

        case VIZIA_BT_MOVE_RIGHT : return Vizia_GetCommandWithState(strdup(" moveright"), state);
        case VIZIA_BT_MOVE_LEFT : return Vizia_GetCommandWithState(strdup(" moveleft"), state);
        case VIZIA_BT_MOVE_BACK : return Vizia_GetCommandWithState(strdup(" back"), state);
        case VIZIA_BT_MOVE_FORWARD : return Vizia_GetCommandWithState(strdup(" forward"), state);
        case VIZIA_BT_TURN_RIGHT : return Vizia_GetCommandWithState(strdup(" right"), state);
        case VIZIA_BT_TURN_LEFT : return Vizia_GetCommandWithState(strdup(" left"), state);
        case VIZIA_BT_LOOK_UP : return Vizia_GetCommandWithState(strdup(" lookup"), state);
        case VIZIA_BT_LOOK_DOWN : return Vizia_GetCommandWithState(strdup(" lookdown"), state);
        case VIZIA_BT_MOVE_UP : return Vizia_GetCommandWithState(strdup(" moveup"), state);
        case VIZIA_BT_MOVE_DOWN : return Vizia_GetCommandWithState(strdup(" movedown"), state);

        case VIZIA_BT_SELECT_WEAPON1 : return strdup("slot 1");
        case VIZIA_BT_SELECT_WEAPON2 : return strdup("slot 2");
        case VIZIA_BT_SELECT_WEAPON3 : return strdup("slot 3");
        case VIZIA_BT_SELECT_WEAPON4 : return strdup("slot 4");
        case VIZIA_BT_SELECT_WEAPON5 : return strdup("slot 5");
        case VIZIA_BT_SELECT_WEAPON6 : return strdup("slot 6");
        case VIZIA_BT_SELECT_WEAPON7 : return strdup("slot 7");

        case VIZIA_BT_WEAPON_NEXT : return strdup("weapnext");
        case VIZIA_BT_WEAPON_PREV : return strdup("weapprev");

        default : return strdup("");
    }
}

char* Vizia_GetCommandWithState(char* command, bool state){
    if(state) command[0] = '+';
    else command[0] = '-';
    return command;
}

void Vizia_InputInit() {
    viziaInputSMRegion = NULL;

    viziaLastInput = new ViziaInputStruct();

    try {
        viziaInputSMRegion = new bip::mapped_region(viziaSM, bip::read_write, Vizia_SMGetInputRegionBeginning(),
                                                    sizeof(ViziaInputStruct));
        viziaInput = static_cast<ViziaInputStruct *>(viziaInputSMRegion->get_address());

        printf("Vizia_InputInit: Input SM region size: %zu, beginnig: %p, end: %p \n",
               viziaInputSMRegion->get_size(), viziaInputSMRegion->get_address(),
               viziaInputSMRegion->get_address() + viziaInputSMRegion->get_size());
    }
    catch (bip::interprocess_exception &ex) {
        printf("Vizia_InputInit: Error creating Input SM");
        Vizia_MQSend(VIZIA_MSG_CODE_DOOM_ERROR);
        Vizia_Command(strdup("exit"));
    }

    for (int i = 0; i < VIZIA_BT_SIZE; ++i) {
        viziaInput->BT[i] = false;
        viziaInput->BT_AVAILABLE[i] = true;
    }

    viziaInput->MS_MAX_X = 90;
    viziaInput->MS_MAX_Y = 90;
    viziaInput->MS_X = 0;
    viziaInput->MS_Y = 0;

}

void Vizia_InputTic(){

    //Vizia_Mouse(viziaInput->MS_X, viziaInput->MS_Y);

    for(int i = 0; i<VIZIA_BT_SIZE; ++i){

        if(viziaInput->BT_AVAILABLE[i]){

            if(viziaInput->BT[i] && Vizia_HasCounterBT(i)){
                int c = Vizia_CounterBT(i);

                if(viziaInput->BT_AVAILABLE[c] && viziaInput->BT[c]){
                    AddCommandString(Vizia_BTToCommand(i, false));
                    AddCommandString(Vizia_BTToCommand(c, false));
                    continue;
                }
                else AddCommandString(Vizia_BTToCommand(c, false));
            }

            if(viziaInput->BT[i] != viziaLastInput->BT[i]){
                AddCommandString(Vizia_BTToCommand(i, viziaInput->BT[i]));
            }
        }
    }

    memcpy ( viziaLastInput, viziaInput, sizeof(ViziaInputStruct) );

    viziaInput->MS_X = 0;
    viziaInput->MS_Y = 0;
}

void Vizia_InputClose(){
    delete(viziaLastInput);
    delete(viziaInputSMRegion);
}