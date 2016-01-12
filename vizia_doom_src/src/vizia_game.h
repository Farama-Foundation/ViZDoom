#ifndef __VIZIA_GAME_H__
#define __VIZIA_GAME_H__

#include <string.h>

#include "dobject.h"
#include "dobjtype.h"
#include "doomtype.h"
#include "name.h"
#include "d_player.h"
//#include "namedef.h"
//#include "sc_man.h"
//#include "sc_man_tokens.h"

int Vizia_CheckItem(FName name);

int Vizia_CheckItem(PClass *type);

const char* Vizia_CheckItemType(PClass *type);

bool Vizia_CheckSelectedWeaponState();

int Vizia_CheckSelectedWeapon();

int Vizia_CheckWeaponAmmo(AWeapon* weapon);

int Vizia_CheckSelectedWeaponAmmo();

int Vizia_CheckSlotAmmo(int slot);

int Vizia_CheckSlotWeapons(int slot);

void Vizia_GameVarsInit();

void Vizia_GameVarsTic();

void Vizia_GameVarsClose();

#endif
