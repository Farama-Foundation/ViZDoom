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

bip::mapped_region *viziaInputSMRegion;
ViziaInputStruct *viziaLastInput;
ViziaInputStruct *viziaInput;

void Vizia_MouseEvent(int x, int y){
    event_t ev = { 0 };

    ev.x = x;
    ev.y = y;

    if (ev.x || ev.y) {
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
        case V_ATTACK : return Vizia_GetCommandWithState(strdup(" attack"), state);
        case V_USE : return Vizia_GetCommandWithState(strdup(" use"), state);
        case V_JUMP : return Vizia_GetCommandWithState(strdup(" jump"), state);
        case V_CROUCH : return Vizia_GetCommandWithState(strdup(" crouch"), state);
        //case V_TURN180 : return strdup("turn180");
        case V_ALTATTACK : return Vizia_GetCommandWithState(strdup(" altattack"), state);
        case V_RELOAD : return Vizia_GetCommandWithState(strdup(" reload"), state);
        case V_ZOOM : return Vizia_GetCommandWithState(strdup(" zoom"), state);

        case V_SPEED : return Vizia_GetCommandWithState(strdup(" speed"), state);
        case V_STRAFE : return Vizia_GetCommandWithState(strdup(" strafe"), state);

        case V_MOVERIGHT : return Vizia_GetCommandWithState(strdup(" moveright"), state);
        case V_MOVELEFT : return Vizia_GetCommandWithState(strdup(" moveleft"), state);
        case V_BACK : return Vizia_GetCommandWithState(strdup(" back"), state);
        case V_FORWARD : return Vizia_GetCommandWithState(strdup(" forward"), state);
        case V_RIGHT : return Vizia_GetCommandWithState(strdup(" right"), state);
        case V_LEFT : return Vizia_GetCommandWithState(strdup(" left"), state);
        case V_LOOKUP : return Vizia_GetCommandWithState(strdup(" lookup"), state);
        case V_LOOKDOWN : return Vizia_GetCommandWithState(strdup(" lookdown"), state);
        case V_MOVEUP : return Vizia_GetCommandWithState(strdup(" moveup"), state);
        //case V_MOVEDOWN : return Vizia_GetCommandWithState(strdup(" showscores"), state);

        case V_WEAPON1 : return strdup("slot 1");
        case V_WEAPON2 : return strdup("slot 2");
        case V_WEAPON3 : return strdup("slot 3");
        case V_WEAPON4 : return strdup("slot 4");
        case V_WEAPON5 : return strdup("slot 5");
        case V_WEAPON6 : return strdup("slot 6");
        case V_WEAPON7 : return strdup("slot 7");

        case V_WEAPONNEXT : return strdup("weapnext");
        case V_WEAPONPREV : return strdup("weapprev");

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
    viziaLastInput = new ViziaInputStruct();

    viziaInputSMRegion = new bip::mapped_region(viziaSM, bip::read_write, Vizia_SMGetInputRegionBeginning(), sizeof(ViziaInputStruct));
    viziaInput = static_cast<ViziaInputStruct *>(viziaInputSMRegion->get_address());

    printf("Input SM region size: %zu, beginnig: %p, end: %p \n",
           viziaInputSMRegion->get_size(), viziaInputSMRegion->get_address(), viziaInputSMRegion->get_address() + viziaInputSMRegion->get_size());
}

void Vizia_InputTic(){

    Vizia_MouseEvent(viziaInput->MS_X, viziaInput->MS_Y);

    //printf("%d %d \n", viziaInput->MS_X, viziaInput->MS_Y);

    for(int i = 0; i<V_BT_SIZE; ++i){
        Vizia_ButtonCommand(i, viziaInput->BT[i], viziaLastInput->BT[i]);
    }

    memcpy ( viziaLastInput, viziaInput, sizeof(ViziaInputStruct) );

    viziaInput->MS_X = 0;
    viziaInput->MS_Y = 0;
    //memset(viziaInput, 0, sizeof(ViziaInputStruct) );
}

void Vizia_InputClose(){
    delete(viziaLastInput);
    delete(viziaInputSMRegion);
}