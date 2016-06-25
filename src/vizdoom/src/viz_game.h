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

#ifndef __VIZDOOM_GAME_H__
#define __VIZDOOM_GAME_H__

#include <string.h>

#include "dobject.h"
#include "dobjtype.h"
#include "doomtype.h"
#include "name.h"
#include "d_player.h"
//#include "namedef.h"
//#include "sc_man.h"
//#include "sc_man_tokens.h"

int VIZ_CheckItem(FName name);

int VIZ_CheckItem(PClass *type);

const char* VIZ_CheckItemType(PClass *type);

bool VIZ_CheckSelectedWeaponState();

int VIZ_CheckSelectedWeapon();

int VIZ_CheckWeaponAmmo(AWeapon* weapon);

int VIZ_CheckSelectedWeaponAmmo();

int VIZ_CheckSlotAmmo(int slot);

int VIZ_CheckSlotWeapons(int slot);

void VIZ_GameVarsInit();

void VIZ_GameVarsTic();

void VIZ_GameVarsClose();

#endif
