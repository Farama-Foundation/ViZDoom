/*
 Copyright (C) 2016 by Wojciech Jaśkowski, Michał Kempka, Grzegorz Runc, Jakub Toczek, Marek Wydmuch

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

#ifndef __VIZ_INPUT_H__
#define __VIZ_INPUT_H__

enum VIZButton{

    VIZ_BT_ATTACK           = 0,
    VIZ_BT_USE              = 1,
    VIZ_BT_JUMP             = 2,
    VIZ_BT_CROUCH           = 3,
    VIZ_BT_TURN180          = 4,
    VIZ_BT_ALTATTACK        = 5,
    VIZ_BT_RELOAD           = 6,
    VIZ_BT_ZOOM             = 7,

    VIZ_BT_SPEED            = 8,
    VIZ_BT_STRAFE           = 9,

    VIZ_BT_MOVE_RIGHT       = 10,
    VIZ_BT_MOVE_LEFT        = 11,
    VIZ_BT_MOVE_BACK        = 12,
    VIZ_BT_MOVE_FORWARD     = 13,
    VIZ_BT_TURN_RIGHT       = 14,
    VIZ_BT_TURN_LEFT        = 15,
    VIZ_BT_LOOK_UP          = 16,
    VIZ_BT_LOOK_DOWN        = 17,
    VIZ_BT_MOVE_UP          = 18,
    VIZ_BT_MOVE_DOWN        = 19,
    VIZ_BT_LAND             = 20,

    VIZ_BT_SELECT_WEAPON1   = 21,
    VIZ_BT_SELECT_WEAPON2   = 22,
    VIZ_BT_SELECT_WEAPON3   = 23,
    VIZ_BT_SELECT_WEAPON4   = 24,
    VIZ_BT_SELECT_WEAPON5   = 25,
    VIZ_BT_SELECT_WEAPON6   = 26,
    VIZ_BT_SELECT_WEAPON7   = 27,
    VIZ_BT_SELECT_WEAPON8   = 28,
    VIZ_BT_SELECT_WEAPON9   = 29,
    VIZ_BT_SELECT_WEAPON0   = 30,

    VIZ_BT_SELECT_NEXT_WEAPON       = 31,
    VIZ_BT_SELECT_PREV_WEAPON       = 32,
    VIZ_BT_DROP_SELECTED_WEAPON     = 33,

    VIZ_BT_ACTIVATE_SELECTED_ITEM   = 34,
    VIZ_BT_SELECT_NEXT_ITEM         = 35,
    VIZ_BT_SELECT_PREV_ITEM         = 36,
    VIZ_BT_DROP_SELECTED_ITEM       = 37,

    VIZ_BT_VIEW_PITCH_AXIS          = 38,
    VIZ_BT_VIEW_ANGLE_AXIS          = 39,
    VIZ_BT_FORWARD_BACKWARD_AXIS    = 40,
    VIZ_BT_LEFT_RIGHT_AXIS          = 41,
    VIZ_BT_UP_DOWN_AXIS             = 42,
};

#define VIZ_BT_CMD_BT_COUNT         38
#define VIZ_BT_AXIS_BT_COUNT        5
#define VIZ_BT_COUNT                43

struct VIZInputState{
    int BT[VIZ_BT_COUNT];
    bool BT_AVAILABLE[VIZ_BT_COUNT];
    int BT_MAX_VALUE[VIZ_BT_AXIS_BT_COUNT];
};

void VIZ_Command(char * cmd);

bool VIZ_CommmandFilter(const char *cmd);

int VIZ_AxisFilter(VIZButton button, int value);

void VIZ_AddAxisBT(VIZButton button, int value);

char* VIZ_AddStateToBTCommmand(char *& cmd, int state);

char* VIZ_BTToCommand(VIZButton button);

void VIZ_ResetDiscontinuousBT();

void VIZ_AddBTCommand(VIZButton button, int state);

void VIZ_InputInit();

void VIZ_InputTic();

void VIZ_InputClose();

#endif
