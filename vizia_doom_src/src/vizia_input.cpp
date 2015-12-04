#include "vizia_input.h"
#include "vizia_shared_memory.h"
#include "vizia_message_queue.h"

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
        case VIZIA_BT_MOVERIGHT : return VIZIA_BT_MOVELEFT;
        case VIZIA_BT_MOVELEFT : return VIZIA_BT_MOVERIGHT;
        case VIZIA_BT_BACK : return VIZIA_BT_FORWARD;
        case VIZIA_BT_FORWARD : return VIZIA_BT_BACK;
        case VIZIA_BT_RIGHT : return VIZIA_BT_LEFT;
        case VIZIA_BT_LEFT : return VIZIA_BT_RIGHT;
        case VIZIA_BT_LOOKUP : return VIZIA_BT_LOOKDOWN;
        case VIZIA_BT_LOOKDOWN : return VIZIA_BT_LOOKUP;
        case VIZIA_BT_MOVEUP : return VIZIA_BT_MOVEDOWN;
        case VIZIA_BT_MOVEDOWN : return VIZIA_BT_MOVEUP;
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

        case VIZIA_BT_MOVERIGHT : return Vizia_GetCommandWithState(strdup(" moveright"), state);
        case VIZIA_BT_MOVELEFT : return Vizia_GetCommandWithState(strdup(" moveleft"), state);
        case VIZIA_BT_BACK : return Vizia_GetCommandWithState(strdup(" back"), state);
        case VIZIA_BT_FORWARD : return Vizia_GetCommandWithState(strdup(" forward"), state);
        case VIZIA_BT_RIGHT : return Vizia_GetCommandWithState(strdup(" right"), state);
        case VIZIA_BT_LEFT : return Vizia_GetCommandWithState(strdup(" left"), state);
        case VIZIA_BT_LOOKUP : return Vizia_GetCommandWithState(strdup(" lookup"), state);
        case VIZIA_BT_LOOKDOWN : return Vizia_GetCommandWithState(strdup(" lookdown"), state);
        case VIZIA_BT_MOVEUP : return Vizia_GetCommandWithState(strdup(" moveup"), state);
        case VIZIA_BT_MOVEDOWN : return Vizia_GetCommandWithState(strdup(" movedown"), state);

        case VIZIA_BT_WEAPON1 : return strdup("slot 1");
        case VIZIA_BT_WEAPON2 : return strdup("slot 2");
        case VIZIA_BT_WEAPON3 : return strdup("slot 3");
        case VIZIA_BT_WEAPON4 : return strdup("slot 4");
        case VIZIA_BT_WEAPON5 : return strdup("slot 5");
        case VIZIA_BT_WEAPON6 : return strdup("slot 6");
        case VIZIA_BT_WEAPON7 : return strdup("slot 7");

        case VIZIA_BT_WEAPONNEXT : return strdup("weapnext");
        case VIZIA_BT_WEAPONPREV : return strdup("weapprev");

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

    Vizia_MouseEvent(viziaInput->MS_X, viziaInput->MS_Y);

    for(int i = 0; i<VIZIA_BT_SIZE; ++i){
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