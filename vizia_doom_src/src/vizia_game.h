#ifndef __VIZIA_GAME_H__
#define __VIZIA_GAME_H__

#include <string.h>

#include "dobject.h"
#include "dobjtype.h"
#include "doomtype.h"
#include "name.h"
//#include "namedef.h"
//#include "sc_man.h"
//#include "sc_man_tokens.h"

int Vizia_CheckItem(FName name);

int Vizia_CheckItem(PClass *type);

const char* Vizia_CheckItemType(PClass *type);

bool Vizia_CheckSelectedWeaponState();

int Vizia_CheckSelectedWeapon();

int Vizia_CheckSelectedWeaponAmmo();

void Vizia_GameVarsInit();

void Vizia_GameVarsTic();

void Vizia_GameVarsUpdate();

void Vizia_GameVarsClose();

#endif
