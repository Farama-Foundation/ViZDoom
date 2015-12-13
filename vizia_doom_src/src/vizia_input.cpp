#include "vizia_input.h"
#include "vizia_message_queue.h"
#include "vizia_shared_memory.h"

#include "d_main.h"
#include "g_game.h"
#include "d_player.h"
#include "d_event.h"
#include "c_bind.h"
#include "c_console.h"
#include "c_dispatch.h"

bip::mapped_region *viziaInputSMRegion = NULL;
ViziaInputStruct *viziaLastInput = NULL;
ViziaInputStruct *viziaInput = NULL;
bool viziaInputInited = false;

EXTERN_CVAR (Bool, vizia_allow_input);

void Vizia_Command(char * command){
    if(strlen(command) >= 1) AddCommandString(command);
}

bool Vizia_CommmandFilter(const char *cmd){

    if(!viziaInputInited || !*vizia_allow_input) return true;

    bool action = false;
    bool state = true;

    if (*cmd == '+'){
        action = true;
        state = true;
    }
    else if(*cmd == '-'){
        action = true;
        state = false;
    }

    const char* beg;
    if(action) beg = cmd+1;
    else beg = cmd;

    for(int i =0; i<VIZIA_BT_SIZE; ++i){
        if(strcmp(beg, Vizia_BTToCommand(i)) == 0){
            if(!viziaInput->BT_AVAILABLE[i]) return false;
            else{
                viziaInput->BT[i] = state;
            }
        }
    }

    return true;
}

char* Vizia_BTToCommand(int button){
    switch(button){
        case VIZIA_BT_ATTACK : return strdup("attack");
        case VIZIA_BT_USE : return strdup("use");
        case VIZIA_BT_JUMP : return strdup("jump");
        case VIZIA_BT_CROUCH : return strdup("crouch");
        case VIZIA_BT_TURN180 : return strdup("turn180");
        case VIZIA_BT_ALTATTACK : return strdup("altattack");
        case VIZIA_BT_RELOAD : return strdup("reload");
        case VIZIA_BT_ZOOM : return strdup("zoom");

        case VIZIA_BT_SPEED : return strdup("speed");
        case VIZIA_BT_STRAFE : return strdup("strafe");

        case VIZIA_BT_MOVE_RIGHT : return strdup("moveright");
        case VIZIA_BT_MOVE_LEFT : return strdup("moveleft");
        case VIZIA_BT_MOVE_BACK : return strdup("back");
        case VIZIA_BT_MOVE_FORWARD : return strdup("forward");
        case VIZIA_BT_TURN_RIGHT : return strdup("right");
        case VIZIA_BT_TURN_LEFT : return strdup("left");
        case VIZIA_BT_LOOK_UP : return strdup("lookup");
        case VIZIA_BT_LOOK_DOWN : return strdup("lookdown");
        case VIZIA_BT_MOVE_UP : return strdup("moveup");
        case VIZIA_BT_MOVE_DOWN : return strdup("movedown");

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

void Vizia_AddBTCommand(int button, bool state){

    switch(button){
        case VIZIA_BT_ATTACK :
        case VIZIA_BT_USE :
        case VIZIA_BT_JUMP :
        case VIZIA_BT_CROUCH :
        case VIZIA_BT_ALTATTACK :
        case VIZIA_BT_RELOAD :
        case VIZIA_BT_ZOOM :
        case VIZIA_BT_SPEED :
        case VIZIA_BT_STRAFE :
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
            if(state) Vizia_Command(strcat(strdup("+"), Vizia_BTToCommand(button)));
            else Vizia_Command(strcat(strdup("-"), Vizia_BTToCommand(button)));
            break;

        case VIZIA_BT_TURN180 :
        case VIZIA_BT_SELECT_WEAPON1 :
        case VIZIA_BT_SELECT_WEAPON2 :
        case VIZIA_BT_SELECT_WEAPON3 :
        case VIZIA_BT_SELECT_WEAPON4 :
        case VIZIA_BT_SELECT_WEAPON5 :
        case VIZIA_BT_SELECT_WEAPON6 :
        case VIZIA_BT_SELECT_WEAPON7 :
        case VIZIA_BT_WEAPON_NEXT :
        case VIZIA_BT_WEAPON_PREV :
            if(state) Vizia_Command(Vizia_BTToCommand(button));
            break;
    }

}

void Vizia_InputInit() {

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

    viziaInputInited = true;
}

void Vizia_InputTic(){

    //Vizia_Mouse(viziaInput->MS_X, viziaInput->MS_Y);
    if(!*vizia_allow_input) {
        for (int i = 0; i < VIZIA_BT_SIZE; ++i) {

            if (viziaInput->BT_AVAILABLE[i]) {

                if (viziaInput->BT[i] && Vizia_HasCounterBT(i)) {
                    int c = Vizia_CounterBT(i);

                    if (viziaInput->BT_AVAILABLE[c] && viziaInput->BT[c]) {
                        Vizia_AddBTCommand(i, false);
                        Vizia_AddBTCommand(c, false);
                        continue;
                    }
                    else Vizia_AddBTCommand(c, false);
                }

                if (viziaInput->BT[i] != viziaLastInput->BT[i]) {
                    Vizia_AddBTCommand(i, viziaInput->BT[i]);
                }
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