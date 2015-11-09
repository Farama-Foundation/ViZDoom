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

int Vizia_CheckEquippedWeapon();

int Vizia_CheckEquippedWeaponAmmo();

void Vizia_GameVarsInit();

void Vizia_UpdateGameVars();

void Vizia_GameVarsClose();

#endif
