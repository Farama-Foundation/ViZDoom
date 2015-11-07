#include "vizia_main.h"
#include "vizia_shared_memory.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>


using namespace boost::interprocess;

int _Vizia_CheckItem(FName name) {
    AInventory *item = viziaPlayer->mo->FindInventory(name);
    if (item == NULL) return 0;
    else return item->Amount;
}

int _Vizia_CheckItem(PClass *type) {
    AInventory *item = viziaPlayer->mo->FindInventory(type);
    if (item == NULL) return 0;
    else return item->Amount;
}

string _Vizia_CheckItemType(PClass *type){

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

int _Vizia_CheckEquippedWeapon(){
    return 0;
}

int _Vizia_CheckEquippedWeaponAmmo(){
    return _Vizia_CheckItem(viziaPlayer->ReadyWeapon->AmmoType1);
}

void _Vizia_UpdateScreen(){

    if(screen->GetWidth() != viziaScreenWidth || screen->GetHeight() != viziaScreenHeight){
        viziaScreenWidth = screen->GetWidth();
        viziaScreenHeight = screen->GetHeight();
        viziaScreenSM->truncate(sizeof(BYTE)*viziaScreenWidth*viziaScreenHeight);
    }

    viziaScreen = screen->GetBuffer();
}

void _Vizia_UpdateGameData(){

    ViziaGameDataSMStruct *newData = new ViziaGameDataSMStruct();

    newData->TIC = gametic;

    newData->SCREEN_HEIGHT = screen->GetHeight();
    newData->SCREEN_WIDTH = screen->GetWidth();

    newData->MAP_FINISHED = 0;

    newData->PLAYER_DEAD = viziaPlayer->playerstate == PST_DEAD;

    newData->PLAYER_KILLCOUNT = viziaPlayer->killcount;
    newData->PLAYER_ITEMCOUNT = viziaPlayer->itemcount;
    newData->PLAYER_SECRETCOUNT = viziaPlayer->secretcount;
    newData->PLAYER_FRAGCOUNT = viziaPlayer->fragcount;

    newData->PLAYER_ONGROUND = viziaPlayer->onground;

    newData->PLAYER_HEALTH = viziaPlayer->mo->health;
    newData->PLAYER_ARMOR = _Vizia_CheckItem(NAME_BasicArmor);

    newData->PLAYER_EQUIPPED_WEAPON_AMMO = _Vizia_CheckEquippedWeaponAmmo();
    newData->PLAYER_EQUIPPED_WEAPON = _Vizia_CheckEquippedWeapon();

    newData->PLAYER_AMMO[0] = _Vizia_CheckItem(NAME_Clip);
    newData->PLAYER_AMMO[1] = _Vizia_CheckItem(NAME_Shell);
    newData->PLAYER_AMMO[2] = _Vizia_CheckItem(NAME_RocketAmmo);
    newData->PLAYER_AMMO[3] = _Vizia_CheckItem(NAME_Cell);

    newData->PLAYER_WEAPON[0] = _Vizia_CheckItem(NAME_Fist) || _Vizia_CheckItem(NAME_Chainsaw);
    newData->PLAYER_WEAPON[1] = (bool)_Vizia_CheckItem(NAME_Pistol);
    newData->PLAYER_WEAPON[2] = _Vizia_CheckItem(NAME_Shotgun) || _Vizia_CheckItem(NAME_SSG);
    newData->PLAYER_WEAPON[3] = (bool)_Vizia_CheckItem(NAME_Chaingun);
    newData->PLAYER_WEAPON[4] = (bool)_Vizia_CheckItem(NAME_Rocket);
    newData->PLAYER_WEAPON[5] = (bool)_Vizia_CheckItem(NAME_Plasma);
    newData->PLAYER_WEAPON[6] = (bool)_Vizia_CheckItem(NAME_BFG);

    //newData.PLAYER_KEY[0] = ?
    //newData.PLAYER_KEY[1] = ?
    //newData.PLAYER_KEY[2] = ?
}

void _Vizia_GetInput(){
    ViziaInputSMStruct *newInput;
    newInput = viziaInput;

    if(newInput->BT_ATTACK){
        
    }
}

void _Vizia_Input(){

}

void Vizia_Init(){

    viziaPlayer = &VIZIA_PLAYER;
    viziaScreenWidth = screen->GetWidth();
    viziaScreenHeight = screen->GetHeight();

    viziaScreenSM = new shared_memory_object(open_or_create, viziaScreenSMName, read_write);
    viziaScreenSM->truncate(sizeof(BYTE)*viziaScreenWidth*viziaScreenHeight);
    mapped_region viziaScreenSMRegion{viziaScreenSM, read_write};
    viziaScreen = static_cast<BYTE*>(viziaScreenSMRegion.get_address());

    viziaGameDataSM = new shared_memory_object(open_or_create, viziaGameDataSMName, read_write);
    viziaGameDataSM->truncate(sizeof(ViziaGameDataSMStruct));
    mapped_region viziaGameDataSMRegion{viziaScreenSM, read_write};
    viziaGameData = static_cast<ViziaGameDataSMStruct*>(viziaGameDataSMRegion.get_address());

    viziaInputSM = new shared_memory_object(open_or_create, viziaInputSMName, read_write);
    viziaInputSM->truncate(sizeof(ViziaInputSMStruct));
    mapped_region viziaInputSMRegion{viziaScreenSM, read_write};
    viziaInput = static_cast<ViziaInputSMStruct*>(viziaInputSMRegion.get_address());
}

void Vizia_Close(){
    shared_memory_object::remove(viziaScreenSMName);
    shared_memory_object::remove(viziaGameDataSMName);
    shared_memory_object::remove(viziaInputSMName);
}

void Vizia_Tic(){

    _Vizia_GetInput();
    _Vizia_UpdateEngineData();
    _Vizia_UpdateScreen();

}

