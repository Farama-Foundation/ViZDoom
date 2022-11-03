/*
 Copyright (C) 2016 by Wojciech Jaśkowski, Michał Kempka, Grzegorz Runc, Jakub Toczek, Marek Wydmuch
 Copyright (C) 2017 - 2022 by Marek Wydmuch, Michał Kempka, Wojciech Jaśkowski, and the respective contributors

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
*/

#include "viz_input.h"
#include "viz_main.h"
#include "viz_defines.h"
#include "viz_message_queue.h"
#include "viz_shared_memory.h"

#include "d_main.h"
#include "g_game.h"
#include "d_net.h"
#include "d_player.h"
#include "d_event.h"
#include "c_dispatch.h"
#include "r_utility.h"

VIZInputState *vizInput = NULL;
bool vizInputInited = false;
float vizLastInputBT[VIZ_BT_COUNT];
unsigned int vizLastInputUpdate[VIZ_BT_COUNT];

EXTERN_CVAR (Int, viz_debug)
EXTERN_CVAR (Bool, viz_allow_input)
EXTERN_CVAR (Bool, viz_nocheat)
EXTERN_CVAR (Bool, viz_cmd_filter)
EXTERN_CVAR (Bool, viz_spectator)

void VIZ_Command(char * cmd){
    AddCommandString(cmd);
}

bool VIZ_CommmandFilter(const char *cmd){

    VIZ_DebugMsg(3, VIZ_FUNC, "allow_input: %d, cmd: %s", *viz_allow_input, cmd);

    if(*viz_spectator && (strcmp(cmd, "+attack") == 0 || strcmp(cmd, "+altattack") == 0)) return false;
    if(!vizInputInited || !*viz_allow_input || (!*viz_cmd_filter && (!*viz_nocheat || *viz_spectator))) return true;

    bool action = false;
    int state = 1;

    if (*cmd == '+'){
        action = true;
        state = 1;
    }
    else if(*cmd == '-'){
        action = true;
        state = 0;
    }

    const char* beg;
    if(action) beg = cmd+1;
    else beg = cmd;

    if (strcmp(beg, "togglemap") == 0 ||
        strcmp(beg, "toggleconsole") == 0 ||
        strcmp(beg, "showscores") == 0 ||
        strcmp(beg, "menu_main") == 0 ||
        strcmp(beg, "menu_help") == 0 ||
        strcmp(beg, "menu_save") == 0 ||
        strcmp(beg, "menu_load") == 0 ||
        strcmp(beg, "menu_options") == 0 ||
        strcmp(beg, "menu_display") == 0 ||
        strcmp(beg, "quicksave") == 0 ||
        strcmp(beg, "menu_endgame") == 0 ||
        strcmp(beg, "togglemessages") == 0 ||
        strcmp(beg, "quickload") == 0 ||
        strcmp(beg, "menu_quit") == 0 ||
        strcmp(beg, "bumpgamma") == 0 ||
        strcmp(beg, "spynext") == 0 ||
        strcmp(beg, "screenshot") == 0 ||
        strcmp(beg, "pause") == 0 ||
        strcmp(beg, "centerview") == 0){
        return false;
    }

    for(int i = 0; i < VIZ_BT_CMD_BT_COUNT; ++i){

        char * ckeckCmd = VIZ_BTToCommand((VIZButton)i);

        if (strcmp(beg, ckeckCmd) == 0){
            if(!vizInput->BT_AVAILABLE[i]) {
                vizInput->BT[i] = 0;
                return false;
            }
            else{
                vizInput->BT[i] = state;
                vizLastInputUpdate[i] = VIZ_TIME;
            }
            //vizInput->CMD_BT[i] = state;
        }

        delete[] ckeckCmd;
    }

    return true;
}

void VIZ_ReadUserCmdState(usercmd_t *ucmd, int player){
    if(vizInputInited && player == consoleplayer) {
        for (size_t i = 0; i < 20; ++i) {
            int bt = 1 << i; // ViZDoom's buttons map to the same values as the engine's buttons enum
            vizInput->CMD_BT[i] = (ucmd->buttons & bt) != 0 ? 1.0 : 0.0;
        }
        vizInput->CMD_BT[VIZ_BT_VIEW_ANGLE_AXIS] = -static_cast<double>(ucmd->yaw)/32768 * 180;
        vizInput->CMD_BT[VIZ_BT_VIEW_PITCH_AXIS] = static_cast<double>(ucmd->pitch)/32768 * 180;
        vizInput->CMD_BT[VIZ_BT_FORWARD_BACKWARD_AXIS] = static_cast<double>(ucmd->forwardmove)/256;
        vizInput->CMD_BT[VIZ_BT_LEFT_RIGHT_AXIS] = static_cast<double>(ucmd->sidemove)/256;
        vizInput->CMD_BT[VIZ_BT_UP_DOWN_AXIS] = static_cast<double>(ucmd->upmove)/256;
    }
}

int VIZ_AxisFilter(VIZButton button, double value){
    if(vizInputInited && value != 0 && button >= VIZ_BT_CMD_BT_COUNT && button < VIZ_BT_COUNT){

        if(!vizInput->BT_AVAILABLE[button]) return 0;
        size_t axis = button - VIZ_BT_CMD_BT_COUNT;
        if(vizInput->BT_MAX_VALUE[axis] != 0){
            double maxValue;
            if(button == VIZ_BT_VIEW_ANGLE_AXIS || button == VIZ_BT_VIEW_PITCH_AXIS)
                maxValue = vizInput->BT_MAX_VALUE[axis]/180.0 * 32768.0;
            else maxValue = vizInput->BT_MAX_VALUE[axis];

            if(std::abs(value) > std::abs(maxValue))
                value = value/std::abs(value) * std::abs(maxValue) + 1;
        }
        if(button == VIZ_BT_VIEW_ANGLE_AXIS || button == VIZ_BT_VIEW_PITCH_AXIS)
            vizInput->BT[button] = value/32768.0 * 180.0;
        else vizInput->BT[button] = value;
        vizLastInputUpdate[button] = VIZ_TIME;
    }
    return static_cast<int>(floor(value));
}

void VIZ_AddAxisBT(VIZButton button, double value){
    if(button == VIZ_BT_VIEW_ANGLE_AXIS || button == VIZ_BT_VIEW_PITCH_AXIS)
        value = value/180.0 * 32768.0;
    int filtredValue = VIZ_AxisFilter(button, value);
    switch(button){
        case VIZ_BT_VIEW_PITCH_AXIS :
            G_AddViewPitch(filtredValue);
            //LocalViewPitch = filtredValue;
            break;
        case VIZ_BT_VIEW_ANGLE_AXIS :
            G_AddViewAngle(filtredValue);
            //LocalViewAngle = filtredValue;
            break;
        case VIZ_BT_FORWARD_BACKWARD_AXIS :
            LocalForward = filtredValue;
            break;
        case VIZ_BT_LEFT_RIGHT_AXIS :
            LocalSide = filtredValue;
            break;
        case VIZ_BT_UP_DOWN_AXIS :
            LocalFly = filtredValue;
            break;
        default:
            break;
    }
}

char* VIZ_BTToCommand(VIZButton button){

    switch(button){
        case VIZ_BT_ATTACK: return strdup("attack");
        case VIZ_BT_USE : return strdup("use");
        case VIZ_BT_JUMP : return strdup("jump");
        case VIZ_BT_CROUCH : return strdup("crouch");
        case VIZ_BT_TURN180 : return strdup("turn180");
        case VIZ_BT_ALTATTACK : return strdup("altattack");
        case VIZ_BT_RELOAD : return strdup("reload");
        case VIZ_BT_ZOOM : return strdup("zoom");
        case VIZ_BT_SPEED : return strdup("speed");
        case VIZ_BT_STRAFE : return strdup("strafe");
        case VIZ_BT_MOVE_RIGHT: return strdup("moveright");
        case VIZ_BT_MOVE_LEFT: return strdup("moveleft");
        case VIZ_BT_MOVE_BACK : return strdup("back");
        case VIZ_BT_MOVE_FORWARD : return strdup("forward");
        case VIZ_BT_TURN_RIGHT : return strdup("right");
        case VIZ_BT_TURN_LEFT : return strdup("left");
        case VIZ_BT_LOOK_UP : return strdup("lookup");
        case VIZ_BT_LOOK_DOWN : return strdup("lookdown");
        case VIZ_BT_MOVE_UP : return strdup("moveup");
        case VIZ_BT_MOVE_DOWN : return strdup("movedown");
        case VIZ_BT_LAND : return strdup("land");

        case VIZ_BT_SELECT_WEAPON1 : return strdup("slot 1");
        case VIZ_BT_SELECT_WEAPON2 : return strdup("slot 2");
        case VIZ_BT_SELECT_WEAPON3 : return strdup("slot 3");
        case VIZ_BT_SELECT_WEAPON4 : return strdup("slot 4");
        case VIZ_BT_SELECT_WEAPON5 : return strdup("slot 5");
        case VIZ_BT_SELECT_WEAPON6 : return strdup("slot 6");
        case VIZ_BT_SELECT_WEAPON7 : return strdup("slot 7");
        case VIZ_BT_SELECT_WEAPON8 : return strdup("slot 8");
        case VIZ_BT_SELECT_WEAPON9 : return strdup("slot 9");
        case VIZ_BT_SELECT_WEAPON0 : return strdup("slot 0");

        case VIZ_BT_SELECT_NEXT_WEAPON : return strdup("weapnext");
        case VIZ_BT_SELECT_PREV_WEAPON : return strdup("weapprev");
        case VIZ_BT_DROP_SELECTED_WEAPON : return strdup("weapdrop");

        case VIZ_BT_ACTIVATE_SELECTED_ITEM : return strdup("invuse");
        case VIZ_BT_SELECT_NEXT_ITEM : return strdup("invnext");
        case VIZ_BT_SELECT_PREV_ITEM : return strdup("invprev");
        case VIZ_BT_DROP_SELECTED_ITEM : return strdup("invdrop");

        case VIZ_BT_VIEW_PITCH_AXIS :
        case VIZ_BT_VIEW_ANGLE_AXIS :
        case VIZ_BT_FORWARD_BACKWARD_AXIS :
        case VIZ_BT_LEFT_RIGHT_AXIS :
        case VIZ_BT_UP_DOWN_AXIS :
        default : return strdup("");
    }
}

void VIZ_ResetDiscontinuousBT(){

    if(vizLastInputUpdate[VIZ_BT_TURN180] < VIZ_TIME) vizInput->BT[VIZ_BT_TURN180] = 0;
    for(size_t i = VIZ_BT_LAND; i < VIZ_BT_COUNT; ++i){
        if(vizLastInputUpdate[i] < VIZ_TIME) vizInput->BT[i] = 0;
    }
}

char* VIZ_AddStateToBTCommmand(char *& cmd, double state){
    size_t cmdLen = strlen(cmd);
    char *stateCmd = new char[cmdLen + 2];
    if (state != 0) stateCmd[0] = '+';
    else stateCmd[0] = '-';
    strncpy(stateCmd + 1, cmd, cmdLen);
    stateCmd[cmdLen + 1] = '\0';

    delete[] cmd;
    cmd = stateCmd;

    return stateCmd;
}

void VIZ_AddBTCommand(VIZButton button, double state){

    char* buttonCmd = VIZ_BTToCommand(button);

    switch(button){
        case VIZ_BT_ATTACK :
        case VIZ_BT_USE :
        case VIZ_BT_JUMP :
        case VIZ_BT_CROUCH :
        case VIZ_BT_ALTATTACK :
        case VIZ_BT_RELOAD :
        case VIZ_BT_ZOOM :
        case VIZ_BT_SPEED :
        case VIZ_BT_STRAFE :
        case VIZ_BT_MOVE_RIGHT :
        case VIZ_BT_MOVE_LEFT :
        case VIZ_BT_MOVE_BACK :
        case VIZ_BT_MOVE_FORWARD :
        case VIZ_BT_TURN_RIGHT :
        case VIZ_BT_TURN_LEFT :
        case VIZ_BT_LOOK_UP :
        case VIZ_BT_LOOK_DOWN :
        case VIZ_BT_MOVE_UP :
        case VIZ_BT_MOVE_DOWN:
            VIZ_AddStateToBTCommmand(buttonCmd, state);
            VIZ_Command(buttonCmd);
            break;

        case VIZ_BT_TURN180 :
        case VIZ_BT_LAND :
        case VIZ_BT_SELECT_WEAPON1 :
        case VIZ_BT_SELECT_WEAPON2 :
        case VIZ_BT_SELECT_WEAPON3 :
        case VIZ_BT_SELECT_WEAPON4 :
        case VIZ_BT_SELECT_WEAPON5 :
        case VIZ_BT_SELECT_WEAPON6 :
        case VIZ_BT_SELECT_WEAPON7 :
        case VIZ_BT_SELECT_WEAPON8 :
        case VIZ_BT_SELECT_WEAPON9 :
        case VIZ_BT_SELECT_WEAPON0 :
        case VIZ_BT_SELECT_NEXT_WEAPON :
        case VIZ_BT_SELECT_PREV_WEAPON :
        case VIZ_BT_DROP_SELECTED_WEAPON :
        case VIZ_BT_ACTIVATE_SELECTED_ITEM :
        case VIZ_BT_SELECT_NEXT_ITEM :
        case VIZ_BT_SELECT_PREV_ITEM :
        case VIZ_BT_DROP_SELECTED_ITEM :
            if (state) VIZ_Command(buttonCmd);
            break;

        case VIZ_BT_VIEW_PITCH_AXIS :
        case VIZ_BT_VIEW_ANGLE_AXIS :
        case VIZ_BT_FORWARD_BACKWARD_AXIS :
        case VIZ_BT_LEFT_RIGHT_AXIS :
        case VIZ_BT_UP_DOWN_AXIS :
            if(state != 0) VIZ_AddAxisBT(button, state);
            break;
    }

    delete[] buttonCmd;
}

void VIZ_InputInit() {

    try {
        VIZSMRegion* inputRegion = &VIZ_SM_INPUTSTATE;
        VIZ_SMCreateRegion(inputRegion, true, VIZ_SMGetRegionOffset(inputRegion), sizeof(VIZInputState));
        vizInput = static_cast<VIZInputState *>(inputRegion->address);

        VIZ_DebugMsg(1, VIZ_FUNC, "inputOffset: %zu, inputSize: %zu", inputRegion->offset, sizeof(VIZInputState));
    }
    catch (bip::interprocess_exception &ex) {
        VIZ_Error(VIZ_FUNC, "Failed to create input.");
    }

    for (size_t i = 0; i < VIZ_BT_COUNT; ++i) {
        vizInput->BT[i] = 0;
        vizInput->BT_AVAILABLE[i] = true;
        vizLastInputBT[i] = 0;
        vizLastInputUpdate[i] = 0;
    }

    for(size_t i = 0; i < VIZ_BT_AXIS_BT_COUNT; ++i){
        vizInput->BT_MAX_VALUE[i] = 0;
    }

    LocalForward = 0;
    LocalSide = 0;
    LocalFly = 0;
    LocalViewAngle = 0;
    LocalViewPitch = 0;

    vizInputInited = true;
}

void VIZ_InputTic(){

    if(!*viz_allow_input) {
        for (size_t i = 0; i < VIZ_BT_CMD_BT_COUNT; ++i) {
            if (vizInput->BT_AVAILABLE[i] && vizInput->BT[i] != vizLastInputBT[i]){
                VIZ_AddBTCommand((VIZButton)i, vizInput->BT[i]);
            }
        }

        for (size_t i = VIZ_BT_CMD_BT_COUNT; i < VIZ_BT_COUNT; ++i) {
            if (vizInput->BT_AVAILABLE[i]) {
                VIZ_AddBTCommand((VIZButton)i, vizInput->BT[i]);
            }
        }
    }
    else if(vizLastUpdate == VIZ_TIME){
        VIZ_ResetDiscontinuousBT();
        D_ProcessEvents ();
    }

    for (size_t i = 0; i < VIZ_BT_COUNT; ++i) {
        vizLastInputBT[i] = vizInput->BT[i];
    }
}

void VIZ_InputClose(){
    VIZ_SMDeleteRegion(&VIZ_SM_INPUTSTATE);
}

void VIZ_PrintInput() {
    printf("input state: tic %d: buttons: (input/cmd)\n", gametic);
    for (size_t i = 0; i < VIZ_BT_CMD_BT_COUNT; ++i) {
        if (vizInput->BT_AVAILABLE[i]) {
            printf("%f/%f ", vizInput->BT[i], vizInput->CMD_BT[i]);
        }
    }
    printf("\n");
}

