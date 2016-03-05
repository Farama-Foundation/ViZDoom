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

int ViZDoom_CheckItem(FName name);

int ViZDoom_CheckItem(PClass *type);

const char* ViZDoom_CheckItemType(PClass *type);

bool ViZDoom_CheckSelectedWeaponState();

int ViZDoom_CheckSelectedWeapon();

int ViZDoom_CheckWeaponAmmo(AWeapon* weapon);

int ViZDoom_CheckSelectedWeaponAmmo();

int ViZDoom_CheckSlotAmmo(int slot);

int ViZDoom_CheckSlotWeapons(int slot);

void ViZDoom_GameVarsInit();

void ViZDoom_GameVarsTic();

void ViZDoom_GameVarsClose();

#endif
