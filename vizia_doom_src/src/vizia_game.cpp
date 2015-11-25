#include "vizia_game.h"
#include "vizia_shared_memory.h"
#include "vizia_screen.h"

#include "info.h"
#include "d_player.h"
#include "d_netinf.h"
#include "g_game.h"
#include "g_level.h"
#include "g_shared/a_pickups.h"
#include "g_shared/a_keys.h"

#include "info.h"
#include "g_game.h"
#include "g_level.h"
#include "c_console.h"
#include "c_dispatch.h"
#include "p_acs.h"


#include <string.h>
#include <stdio.h>
#include <stdlib.h>

EXTERN_CVAR (Int, vizia_screen_format)

#define VIZIA_PLAYER players[consoleplayer]
player_t *viziaPlayer;

bip::mapped_region *viziaGameVarsSMRegion;
ViziaGameVarsStruct *viziaGameVars;

int Vizia_CheckItem(FName name) {
    if(viziaPlayer->mo != NULL) {
        AInventory *item = viziaPlayer->mo->FindInventory(name);
        if(item != NULL) return item->Amount;
    }
    return 0;
}

int Vizia_CheckItem(const PClass *type) {
    if(viziaPlayer->mo != NULL) {
        AInventory *item = viziaPlayer->mo->FindInventory(type);
        if (item != NULL) item->Amount;
    }
    return 0;
}

const char* Vizia_CheckItemType(const PClass *type){

    if (type->ParentClass == RUNTIME_CLASS(AAmmo)) {
        return "AMMO";
    }
    else if (type->ParentClass == RUNTIME_CLASS(AKey)){
        return "KEY";
    }
    else if (type->ParentClass == RUNTIME_CLASS(AArmor)){
        return "ARMOR";
    }
    else if (type->ParentClass == RUNTIME_CLASS(AWeapon)){
        return "WEAPON";
    }
    return "UNKNOWN";
}

int Vizia_CheckEquippedWeapon(){
    return 0;
}

int Vizia_CheckEquippedWeaponAmmo(){
    //return Vizia_CheckItem(viziaPlayer->ReadyWeapon->AmmoType1);
    return 0;
}

void Vizia_GameVarsInit(){

    viziaPlayer = &players[consoleplayer];

    viziaGameVarsSMRegion = new bip::mapped_region(viziaSM, bip::read_write, Vizia_SMGetGameVarsRegionBeginning(), sizeof(ViziaGameVarsStruct));
    viziaGameVars = static_cast<ViziaGameVarsStruct *>(viziaGameVarsSMRegion->get_address());

    printf("Vizia_GameVarsInit: GameVars SM region size: %zu, beginnig: %p, end: %p \n",
           viziaGameVarsSMRegion->get_size(), viziaGameVarsSMRegion->get_address(), viziaGameVarsSMRegion->get_address() + viziaGameVarsSMRegion->get_size());
}

void Vizia_UpdateGameVars(){

    viziaGameVars->GAME_TIC = gametic;

    viziaGameVars->SCREEN_WIDTH = viziaScreenWidth;
    viziaGameVars->SCREEN_HEIGHT = viziaScreenHeight;
    viziaGameVars->SCREEN_PITCH = viziaScreenPitch;
    viziaGameVars->SCREEN_SIZE = viziaScreenSize;
    viziaGameVars->SCREEN_FORMAT = *vizia_screen_format;

    viziaGameVars->MAP_START_TIC = level.starttime;
    viziaGameVars->MAP_TIC = level.maptime;

    viziaGameVars->MAP_KILLCOUNT = level.killed_monsters;
    viziaGameVars->MAP_ITEMCOUNT = level.found_items;
    viziaGameVars->MAP_SECRETCOUNT = level.found_secrets;

    viziaGameVars->MAP_REWARD = ACS_GlobalVars[0];

    viziaGameVars->MAP_END = false;

    if(viziaPlayer->mo) viziaGameVars->PLAYER_DEAD = viziaPlayer->playerstate == PST_DEAD || viziaPlayer->mo->health <= 0;
    else viziaGameVars->PLAYER_DEAD = viziaPlayer->playerstate == PST_DEAD || viziaPlayer->health <= 0;

    viziaGameVars->PLAYER_KILLCOUNT = viziaPlayer->killcount;
    viziaGameVars->PLAYER_ITEMCOUNT = viziaPlayer->itemcount;
    viziaGameVars->PLAYER_SECRETCOUNT = viziaPlayer->secretcount;
    viziaGameVars->PLAYER_FRAGCOUNT = viziaPlayer->fragcount;

    viziaGameVars->PLAYER_ONGROUND = viziaPlayer->onground;

    if(viziaPlayer->mo) viziaGameVars->PLAYER_HEALTH = viziaPlayer->mo->health;
    else viziaGameVars->PLAYER_HEALTH = viziaPlayer->health;

    viziaGameVars->PLAYER_ARMOR = Vizia_CheckItem(NAME_BasicArmor);

    viziaGameVars->PLAYER_SELECTED_WEAPON_AMMO = Vizia_CheckEquippedWeaponAmmo();
    viziaGameVars->PLAYER_SELECTED_WEAPON = Vizia_CheckEquippedWeapon();

    viziaGameVars->PLAYER_AMMO[0] = Vizia_CheckItem(NAME_Clip);
    viziaGameVars->PLAYER_AMMO[1] = Vizia_CheckItem(NAME_Shell);
    viziaGameVars->PLAYER_AMMO[2] = Vizia_CheckItem(NAME_RocketAmmo);
    viziaGameVars->PLAYER_AMMO[3] = Vizia_CheckItem(NAME_Cell);

    viziaGameVars->PLAYER_WEAPON[0] = Vizia_CheckItem(NAME_Fist) || Vizia_CheckItem(NAME_Chainsaw);
    viziaGameVars->PLAYER_WEAPON[1] = (bool)Vizia_CheckItem(NAME_Pistol);
    viziaGameVars->PLAYER_WEAPON[2] = Vizia_CheckItem(NAME_Shotgun) || Vizia_CheckItem(NAME_SSG);
    viziaGameVars->PLAYER_WEAPON[3] = (bool)Vizia_CheckItem(NAME_Chaingun);
    viziaGameVars->PLAYER_WEAPON[4] = (bool)Vizia_CheckItem(NAME_Rocket);
    viziaGameVars->PLAYER_WEAPON[5] = (bool)Vizia_CheckItem(NAME_Plasma);
    viziaGameVars->PLAYER_WEAPON[6] = (bool)Vizia_CheckItem(NAME_BFG);

    viziaGameVars->PLAYER_KEY[0] = 0;
    viziaGameVars->PLAYER_KEY[1] = 0;
    viziaGameVars->PLAYER_KEY[2] = 0;
}

void Vizia_GameVarsClose(){
    delete(viziaGameVarsSMRegion);
}



