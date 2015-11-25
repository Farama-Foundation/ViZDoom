#include "vizia_input.h"
#include "vizia_shared_memory.h"

#include "d_main.h"
#include "g_game.h"
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

int Vizia_CounterBT(int button){
    switch(button){
        case A_MOVERIGHT : return A_MOVELEFT;
        case A_MOVELEFT : return A_MOVERIGHT;
        case A_BACK : return A_FORWARD;
        case A_FORWARD : return A_BACK;
        case A_RIGHT : return A_LEFT;
        case A_LEFT : return A_RIGHT;
        case A_LOOKUP : return A_LOOKDOWN;
        case A_LOOKDOWN : return A_LOOKUP;
        case A_MOVEUP : return A_MOVEDOWN;
        case A_MOVEDOWN : return A_MOVEUP;
        default : return -1;
    }
}

char* Vizia_BTToCommand(int button, bool state){
    switch(button){
        case A_ATTACK : return Vizia_GetCommandWithState(strdup(" attack"), state);
        case A_USE : return Vizia_GetCommandWithState(strdup(" use"), state);
        case A_JUMP : return Vizia_GetCommandWithState(strdup(" jump"), state);
        case A_CROUCH : return Vizia_GetCommandWithState(strdup(" crouch"), state);
        case A_TURN180 : return strdup("turn180");
        case A_ALTATTACK : return Vizia_GetCommandWithState(strdup(" altattack"), state);
        case A_RELOAD : return Vizia_GetCommandWithState(strdup(" reload"), state);
        case A_ZOOM : return Vizia_GetCommandWithState(strdup(" zoom"), state);

        case A_SPEED : return Vizia_GetCommandWithState(strdup(" speed"), state);
        case A_STRAFE : return Vizia_GetCommandWithState(strdup(" strafe"), state);

        case A_MOVERIGHT : return Vizia_GetCommandWithState(strdup(" moveright"), state);
        case A_MOVELEFT : return Vizia_GetCommandWithState(strdup(" moveleft"), state);
        case A_BACK : return Vizia_GetCommandWithState(strdup(" back"), state);
        case A_FORWARD : return Vizia_GetCommandWithState(strdup(" forward"), state);
        case A_RIGHT : return Vizia_GetCommandWithState(strdup(" right"), state);
        case A_LEFT : return Vizia_GetCommandWithState(strdup(" left"), state);
        case A_LOOKUP : return Vizia_GetCommandWithState(strdup(" lookup"), state);
        case A_LOOKDOWN : return Vizia_GetCommandWithState(strdup(" lookdown"), state);
        case A_MOVEUP : return Vizia_GetCommandWithState(strdup(" moveup"), state);
        case A_MOVEDOWN : return Vizia_GetCommandWithState(strdup(" movedown"), state);

        case A_WEAPON1 : return strdup("slot 1");
        case A_WEAPON2 : return strdup("slot 2");
        case A_WEAPON3 : return strdup("slot 3");
        case A_WEAPON4 : return strdup("slot 4");
        case A_WEAPON5 : return strdup("slot 5");
        case A_WEAPON6 : return strdup("slot 6");
        case A_WEAPON7 : return strdup("slot 7");

        case A_WEAPONNEXT : return strdup("weapnext");
        case A_WEAPONPREV : return strdup("weapprev");

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
        if(state) AddCommandString(Vizia_BTToCommand(Vizia_CounterBT(button), !state));
        AddCommandString(Vizia_BTToCommand(button, state));
    }

}

void Vizia_InputInit(){
    viziaLastInput = new ViziaInputStruct();

    viziaInputSMRegion = new bip::mapped_region(viziaSM, bip::read_write, Vizia_SMGetInputRegionBeginning(), sizeof(ViziaInputStruct));
    viziaInput = static_cast<ViziaInputStruct *>(viziaInputSMRegion->get_address());

    for(int i = 0; i < A_BT_SIZE; ++i) {
        viziaInput->BT[i] = false;
        viziaInput->BT_AVAILABLE[i] = true;
    }

    viziaInput->MS_MAX_X = 90;
    viziaInput->MS_MAX_Y = 90;
    viziaInput->MS_X = 0;
    viziaInput->MS_Y = 0;

    printf("Vizia_InputInit: Input SM region size: %zu, beginnig: %p, end: %p \n",
           viziaInputSMRegion->get_size(), viziaInputSMRegion->get_address(), viziaInputSMRegion->get_address() + viziaInputSMRegion->get_size());
}

void Vizia_InputTic(){

    Vizia_MouseEvent(viziaInput->MS_X, viziaInput->MS_Y);

    for(int i = 0; i<A_BT_SIZE; ++i){
        if(viziaInput->BT_AVAILABLE[i]) Vizia_ButtonCommand(i, viziaInput->BT[i], viziaLastInput->BT[i]);
    }

    memcpy ( viziaLastInput, viziaInput, sizeof(ViziaInputStruct) );

    viziaInput->MS_X = 0;
    viziaInput->MS_Y = 0;
}

void Vizia_InputClose(){
    delete(viziaLastInput);
    delete(viziaInputSMRegion);
}

// OLD EVENTS CODE
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