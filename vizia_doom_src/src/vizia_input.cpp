#include "vizia_input.h"
#include "vizia_shared_memory.h"

#include "d_main.h"
#include "d_event.h"
#include "c_bind.h"
#include "c_console.h"
#include "c_dispatch.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

ViziaInputSMStruct *viziaLastInput;
ViziaInputSMStruct *viziaInput;
//shared_memory_object *viziaInputSM;

void Vizia_MouseEvent(int x, int y){
    event_t ev = { 0 };

    ev.x = x;
    ev.y = y;

    if (ev.x || ev.y)
    {
        ev.type = EV_Mouse;
        D_PostEvent(&ev);
    }
}

void Vizia_ButtonEvent(int button, bool state, bool oldState){

    if(state != oldState){
        event_t ev = { 0 };

        if(state == true) ev.type = EV_KeyDown;
        else ev.type = EV_KeyUp;

        ev.data1 = 1<<button;

        D_PostEvent(&ev);

    }
}

char* Vizia_BTToCommand(int button, bool state){
    switch(button){
        case VBT_ATTACK : return Vizia_GetCommandWithState(strdup(" attack"), state);
        case VBT_USE : return Vizia_GetCommandWithState(strdup(" use"), state);
        case VBT_JUMP : return Vizia_GetCommandWithState(strdup(" jump"), state);
        case VBT_CROUCH : return Vizia_GetCommandWithState(strdup(" crouch"), state);
        //case VBT_TURN180 : return strdup("turn180");
        case VBT_ALTATTACK : return Vizia_GetCommandWithState(strdup(" altattack"), state);
        case VBT_RELOAD : return Vizia_GetCommandWithState(strdup(" reload"), state);
        case VBT_ZOOM : return Vizia_GetCommandWithState(strdup(" zoom"), state);

        case VBT_SPEED : return Vizia_GetCommandWithState(strdup(" speed"), state);
        case VBT_STRAFE : return Vizia_GetCommandWithState(strdup(" strafe"), state);

        case VBT_MOVERIGHT : return Vizia_GetCommandWithState(strdup(" moveright"), state);
        case VBT_MOVELEFT : return Vizia_GetCommandWithState(strdup(" moveleft"), state);
        case VBT_BACK : return Vizia_GetCommandWithState(strdup(" back"), state);
        case VBT_FORWARD : return Vizia_GetCommandWithState(strdup(" forward"), state);
        case VBT_RIGHT : return Vizia_GetCommandWithState(strdup(" right"), state);
        case VBT_LEFT : return Vizia_GetCommandWithState(strdup(" left"), state);
        case VBT_LOOKUP : return Vizia_GetCommandWithState(strdup(" lookup"), state);
        case VBT_LOOKDOWN : return Vizia_GetCommandWithState(strdup(" lookdown"), state);
        case VBT_MOVEUP : return Vizia_GetCommandWithState(strdup(" moveup"), state);
        //case VBT_MOVEDOWN : return Vizia_GetCommandWithState(strdup(" showscores"), state);

        case VBT_WEAPON1 : return strdup("slot 1");
        case VBT_WEAPON2 : return strdup("slot 2");
        case VBT_WEAPON3 : return strdup("slot 3");
        case VBT_WEAPON4 : return strdup("slot 4");
        case VBT_WEAPON5 : return strdup("slot 5");
        case VBT_WEAPON6 : return strdup("slot 6");
        case VBT_WEAPON7 : return strdup("slot 7");

        case VBT_WEAPONNEXT : return strdup("weapnext");
        case VBT_WEAPONPREV : return strdup("weapprev");

        default : return strdup("");
    }
}

char* Vizia_GetCommandWithState(char* command, bool state){
    if(state) command[0] = '+';
    else command[0] = '-';
    return command;
}

void Vizia_ButtonCommand(int button, bool state, bool oldState){

    if(state != oldState){
        AddCommandString(Vizia_BTToCommand(button, state));
    }

}

void Vizia_InputInit(){
    viziaLastInput = new ViziaInputSMStruct();

    //viziaInputSM = new shared_memory_object(create_only, VIZIA_INPUT_SM_NAME, read_write);
    //viziaInputSM->truncate(sizeof(ViziaInputSMStruct));
    //mapped_region viziaInputSMRegion(viziaInputSM, read_write);

    mapped_region viziaInputSMRegion(viziaSM, read_write, Vizia_SMGetInputRegionBeginning(), sizeof(ViziaInputSMStruct));
    viziaInput = static_cast<ViziaInputSMStruct *>(viziaInputSMRegion.get_address());
}

void Vizia_InputTic(){

    Vizia_MouseEvent(viziaInput->MS_X, viziaInput->MS_Y);

    //printf("%d %d \n", viziaInput->MS_X, viziaInput->MS_Y);

    for(int i = 0; i<VBT_SIZE; ++i){
        Vizia_ButtonCommand(i, viziaInput->BT[i], viziaLastInput->BT[i]);
    }

    memcpy ( viziaLastInput, viziaInput, sizeof(ViziaInputSMStruct) );

    viziaInput->MS_X = 0;
    viziaInput->MS_Y = 0;
    //memset(viziaInput, 0, sizeof(ViziaInputSMStruct) );
}

void Vizia_InputClose(){
    //shared_memory_object::remove(VIZIA_INPUT_SM_NAME);
    delete(viziaLastInput);
}