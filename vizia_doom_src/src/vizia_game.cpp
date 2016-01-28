#include "vizia_game.h"
#include "vizia_defines.h"
#include "vizia_shared_memory.h"
#include "vizia_message_queue.h"
#include "vizia_screen.h"

#include "d_netinf.h"
#include "d_event.h"
#include "g_game.h"
#include "g_level.h"
#include "g_shared/a_pickups.h"
#include "g_shared/a_keys.h"
#include "c_console.h"
#include "c_dispatch.h"
#include "p_acs.h"


#include <string.h>
#include <stdio.h>
#include <stdlib.h>

EXTERN_CVAR (Int, vizia_screen_format)

#define VIZIA_PLAYER players[consoleplayer]
player_t *viziaPlayer;

bip::mapped_region *viziaGameVarsSMRegion = NULL;
ViziaGameVarsStruct *viziaGameVars = NULL;

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
        if (item != NULL) return item->Amount;
    }
    return 0;
}

int Vizia_CheckWeaponAmmo(AWeapon* weapon){
    if(weapon != NULL) return Vizia_CheckItem(weapon->AmmoType1);
    return -1;
}

int Vizia_CheckSelectedWeapon(){

    if(viziaPlayer->ReadyWeapon == NULL) return -1;

    const PClass *type1 = viziaPlayer->ReadyWeapon->GetClass();
    if(type1 == NULL) return -1;

    for(int i=0; i< VIZIA_GV_SLOTS_SIZE; ++i){
        for(int j = 0; j < viziaPlayer->weapons.Slots[i].Size(); ++j){
            const PClass *type2 = viziaPlayer->weapons.Slots[i].GetWeapon(j);
            //if(strcmp(type1->TypeName.GetChars(), type2->TypeName.GetChars()) == 0) return i;
            if(type1 == type2) return i;
        }
    }

    return -1;
}

int Vizia_CheckSelectedWeaponAmmo(){
    return Vizia_CheckWeaponAmmo(viziaPlayer->ReadyWeapon);
}

int Vizia_CheckSlotAmmo(int slot){
    if(viziaPlayer->weapons.Slots[slot].Size() <= 0) return 0;

    const PClass *typeWeapon = viziaPlayer->weapons.Slots[slot].GetWeapon(0);
    AWeapon *weapon = (AWeapon*) typeWeapon->CreateNew();
    //AWeapon *weapon = (AWeapon*)viziaPlayer->mo->FindInventory(type);
    if (weapon != NULL){
        const PClass *typeAmmo = weapon->AmmoType1;
        weapon->Destroy();
        return Vizia_CheckItem(typeAmmo);
    }
    else return 0;
}

int Vizia_CheckSlotWeapons(int slot){
    int inSlot = 0;
    for(int i = 0; i < viziaPlayer->weapons.Slots[slot].Size(); ++i){
        const PClass *type = viziaPlayer->weapons.Slots[slot].GetWeapon(i);
        inSlot += Vizia_CheckItem(type);
    }
    return inSlot;
}

void Vizia_GameVarsInit(){

    viziaPlayer = &players[consoleplayer];
    try {
        viziaGameVarsSMRegion = new bip::mapped_region(viziaSM, bip::read_write, sizeof(ViziaInputStruct), sizeof(ViziaGameVarsStruct));
        viziaGameVars = static_cast<ViziaGameVarsStruct *>(viziaGameVarsSMRegion->get_address());
    }
    catch(bip::interprocess_exception &ex){
        printf("Vizia_GameVarsInit: Error GameVars SM");
        Vizia_MQSend(VIZIA_MSG_CODE_DOOM_ERROR);
        exit(1);
    }
}

void Vizia_GameVarsTic(){
    viziaGameVars->GAME_TIC = gametic;
    viziaGameVars->GAME_SEED = rngseed;
    viziaGameVars->GAME_STATIC_SEED = staticrngseed;

    viziaGameVars->SCREEN_WIDTH = viziaScreenWidth;
    viziaGameVars->SCREEN_HEIGHT = viziaScreenHeight;
    viziaGameVars->SCREEN_PITCH = viziaScreenPitch;
    viziaGameVars->SCREEN_SIZE = viziaScreenSize;
    viziaGameVars->SCREEN_FORMAT = *vizia_screen_format;

    viziaGameVars->MAP_START_TIC = level.starttime;
    viziaGameVars->MAP_TIC = level.maptime;

    viziaGameVars->MAP_REWARD = ACS_GlobalVars[0];

    for(int i = 0; i < VIZIA_GV_USER_SIZE; ++i){
        viziaGameVars->MAP_USER_VARS[i] = ACS_GlobalVars[i+1];
    }

    viziaGameVars->MAP_END = gamestate != GS_LEVEL || gameaction == ga_completed;

    if(viziaPlayer->mo) viziaGameVars->PLAYER_DEAD = viziaPlayer->playerstate == PST_DEAD || viziaPlayer->mo->health <= 0;
    else viziaGameVars->PLAYER_DEAD = viziaPlayer->playerstate == PST_DEAD || viziaPlayer->health <= 0;

    viziaGameVars->MAP_KILLCOUNT = level.killed_monsters;
    viziaGameVars->MAP_ITEMCOUNT = level.found_items;
    viziaGameVars->MAP_SECRETCOUNT = level.found_secrets;

    viziaGameVars->PLAYER_KILLCOUNT = viziaPlayer->killcount;
    viziaGameVars->PLAYER_ITEMCOUNT = viziaPlayer->itemcount;
    viziaGameVars->PLAYER_SECRETCOUNT = viziaPlayer->secretcount;
    viziaGameVars->PLAYER_FRAGCOUNT = viziaPlayer->fragcount;

    viziaGameVars->PLAYER_ATTACK_READY = (viziaPlayer->WeaponState & WF_WEAPONREADY);
    viziaGameVars->PLAYER_ALTATTACK_READY = (viziaPlayer->WeaponState & WF_WEAPONREADYALT);
    viziaGameVars->PLAYER_ON_GROUND = viziaPlayer->onground;

    if (viziaPlayer->mo) viziaGameVars->PLAYER_HEALTH = viziaPlayer->mo->health;
    else viziaGameVars->PLAYER_HEALTH = viziaPlayer->health;

    viziaGameVars->PLAYER_ARMOR = Vizia_CheckItem(NAME_BasicArmor);
    //TO DO? support for other types of armor

    viziaGameVars->PLAYER_SELECTED_WEAPON_AMMO = Vizia_CheckSelectedWeaponAmmo();
    viziaGameVars->PLAYER_SELECTED_WEAPON = Vizia_CheckSelectedWeapon();

    for (int i = 0; i < VIZIA_GV_SLOTS_SIZE; ++i) {
        viziaGameVars->PLAYER_AMMO[i] = Vizia_CheckSlotAmmo(i);
        viziaGameVars->PLAYER_WEAPON[i] = Vizia_CheckSlotWeapons(i);
    }
}

void Vizia_GameVarsClose(){
    delete(viziaGameVarsSMRegion);
}



