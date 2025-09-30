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

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <sstream>

#include "viz_game.h"

#include "viz_defines.h"
#include "viz_depth.h"
#include "viz_labels.h"
#include "viz_main.h"
#include "viz_buffers.h"
#include "viz_shared_memory.h"
#include "viz_version.h"
#include "viz_doom_classes.h"

#include "c_dispatch.h"
#include "d_netinf.h"
#include "d_event.h"
#include "d_dehacked.h"
#include "g_game.h"
#include "i_sound.h"
#include "p_acs.h"
#include "p_setup.h"

EXTERN_CVAR (Int, viz_debug)
EXTERN_CVAR (Bool, viz_nocheat)
EXTERN_CVAR (Int, viz_screen_format)
EXTERN_CVAR (Bool, viz_depth)
EXTERN_CVAR (Bool, viz_labels)
EXTERN_CVAR (Bool, viz_automap)
EXTERN_CVAR (Bool, viz_objects)
EXTERN_CVAR (Bool, viz_sectors)
EXTERN_CVAR (Bool, viz_soft_audio)
EXTERN_CVAR (Bool, viz_loop_map)
EXTERN_CVAR (Bool, viz_override_player)
EXTERN_CVAR (Bool, viz_spectator)
EXTERN_CVAR (Int, viz_afk_timeout)
EXTERN_CVAR (Float, timelimit)
EXTERN_CVAR (Bool, viz_notifications)
EXTERN_CVAR (Int, viz_notifications_tics)
EXTERN_CVAR (String, snd_backend)

VIZGameState *vizGameStateSM = NULL;
VIZPlayerLogger vizPlayerLogger[VIZ_MAX_PLAYERS];
unsigned int vizUniqueObjectsCount = 0;
std::vector<VIZTicNotifications> vizNotifications;

/* Logger functions */
/*--------------------------------------------------------------------------------------------------------------------*/

void VIZ_LogNotification(int gametic, const char *message){
    if(!*viz_notifications || gamestate != GS_LEVEL) return;

    if(vizNotifications.empty() || vizNotifications.back().gametic != gametic) {
        vizNotifications.push_back({gametic, {message}});
    } else {
        vizNotifications.back().messages.push_back(message);
    }
}

void VIZ_LogDmg(AActor *target, AActor *inflictor, AActor *source, int amount){
    if(amount < 0) return;
    if(amount > 1000) return; // ignore things like telefrags

    //printf("%s - %d -> %s (%d)\n", source->GetClass()->TypeName.GetChars(), amount, target->GetClass()->TypeName.GetChars(), target->health);

    if(netgame || multiplayer) {
        for (size_t i = 0; i < VIZ_MAX_PLAYERS; ++i) {
            if (players[i].mo == target) {
                vizPlayerLogger[i].dmgTaken += amount;
                ++vizPlayerLogger[i].hitsTaken;

                if (target == source) { // || target == inflictor){
                    vizPlayerLogger[i].selfInflictedDamege += amount;
                    ++vizPlayerLogger[i].selfHitCount;
                }
            } else if (players[i].mo == source) { //vizPlayerLogger[i].actor == inflictor){
                vizPlayerLogger[i].dmgCount += amount;
                ++vizPlayerLogger[i].hitCount;
            }
        }
    }
    else{
        if(VIZ_PLAYER.mo == target){
            vizPlayerLogger[VIZ_PLAYER_NUM].dmgTaken += amount;
            ++vizPlayerLogger[VIZ_PLAYER_NUM].hitsTaken;

            if(target == source){ // || target == inflictor){
                vizPlayerLogger[VIZ_PLAYER_NUM].selfInflictedDamege += amount;
                ++vizPlayerLogger[VIZ_PLAYER_NUM].selfHitCount;
            }

            //printf("dt: %d, ht: %d, dc: %d, hc: %d\n", vizPlayerLogger[VIZ_PLAYER_NUM].dmgTaken, vizPlayerLogger[VIZ_PLAYER_NUM].hitsTaken, vizPlayerLogger[VIZ_PLAYER_NUM].dmgCount, vizPlayerLogger[VIZ_PLAYER_NUM].hitCount);
        }

        else if(VIZ_PLAYER.mo == source){
            vizPlayerLogger[VIZ_PLAYER_NUM].dmgCount += amount;
            ++vizPlayerLogger[VIZ_PLAYER_NUM].hitCount;

            //printf("dt: %d, ht: %d, dc: %d, hc: %d\n", vizPlayerLogger[VIZ_PLAYER_NUM].dmgTaken, vizPlayerLogger[VIZ_PLAYER_NUM].hitsTaken, vizPlayerLogger[VIZ_PLAYER_NUM].dmgCount, vizPlayerLogger[VIZ_PLAYER_NUM].hitCount);
        }
    }
}


/* Helper functions */
/*--------------------------------------------------------------------------------------------------------------------*/

template<typename T>
inline T VIZ_FixedToReal(fixed_t fixed){
    return static_cast<T>(fixed) / 65536.0;
}

inline float VIZ_FixedToFloat(fixed_t fixed){
    return static_cast<float>(fixed) / 65536.0;
}

inline double VIZ_FixedToDouble(fixed_t fixed){
    return static_cast<double>(fixed) / 65536.0;
}

template<typename T>
inline T VIZ_AngleToReal(angle_t angle){
    return static_cast<T>(angle) / ANGLE_MAX * 360.0;
}

inline float  VIZ_AngleToFloat(angle_t angle) {
    return static_cast<float>(angle) / ANGLE_MAX * 360.0;
}

inline double VIZ_AngleToDouble(angle_t angle) {
    return static_cast<double>(angle) / ANGLE_MAX * 360.0;
}

template<typename T>
inline T VIZ_PitchToReal(fixed_t pitch) {
    return static_cast<T>(pitch) / 32768.0 * 180.0 / 65536.0;
}

inline float VIZ_PitchToFloat(fixed_t pitch) {
    return static_cast<float>(pitch) / 32768.0 * 180.0 / 65536.0;
}

inline double VIZ_PitchToDouble(fixed_t pitch) {
    return static_cast<double>(pitch) / 32768.0 * 180.0 / 65536.0;
}

int VIZ_CheckItem(FName name) {
    if(VIZ_PLAYER.mo != NULL) {
        AInventory *item = VIZ_PLAYER.mo->FindInventory(name);
        if(item != NULL) return item->Amount;
    }
    return 0;
}

int VIZ_CheckItem(const PClass *type) {
    if(VIZ_PLAYER.mo != NULL) {
        AInventory *item = VIZ_PLAYER.mo->FindInventory(type);
        if (item != NULL) return item->Amount;
    }
    return 0;
}

int VIZ_CheckWeaponAmmo(AWeapon* weapon){
    if(weapon != NULL) return VIZ_CheckItem(weapon->AmmoType1);
    return -1;
}

int VIZ_CheckSelectedWeapon(){

    if(VIZ_PLAYER.ReadyWeapon == NULL) return -1;

    const PClass *type1 = VIZ_PLAYER.ReadyWeapon->GetClass();
    if(type1 == NULL) return -1;

    for(int i=0; i< VIZ_GV_SLOTS_SIZE; ++i){
        for(int j = 0; j < VIZ_PLAYER.weapons.Slots[i].Size(); ++j){
            const PClass *type2 = VIZ_PLAYER.weapons.Slots[i].GetWeapon(j);
            //if(strcmp(type1->TypeName.GetChars(), type2->TypeName.GetChars()) == 0) return i;
            if(type1 == type2) return i;
        }
    }

    return -1;
}

int VIZ_CheckSelectedWeaponAmmo(){
    return VIZ_CheckWeaponAmmo(VIZ_PLAYER.ReadyWeapon);
}

int VIZ_CheckSlotAmmo(unsigned int slot){
    if(VIZ_PLAYER.weapons.Slots[slot].Size() <= 0) return 0;

    const PClass *typeWeapon = VIZ_PLAYER.weapons.Slots[slot].GetWeapon(0);
    AWeapon *weapon = (AWeapon*) typeWeapon->CreateNew();
    //AWeapon *weapon = (AWeapon*)VIZ_PLAYER.mo->FindInventory(type);
    if (weapon != NULL){
        const PClass *typeAmmo = weapon->AmmoType1;
        weapon->Destroy();
        return VIZ_CheckItem(typeAmmo);
    }
    else return 0;
}

int VIZ_CheckSlotWeapons(unsigned int slot){
    int inSlot = 0;
    for(int i = 0; i < VIZ_PLAYER.weapons.Slots[slot].Size(); ++i){
        const PClass *type = VIZ_PLAYER.weapons.Slots[slot].GetWeapon(i);
        inSlot += VIZ_CheckItem(type);
    }
    return inSlot;
}

void VIZ_CopyActorName(AActor* actor, char* name) {
    //if(actor->health <= 0 || (actor->flags & MF_CORPSE) || (actor->flags6 & MF6_KILLED)) {
    if ((actor->flags & MF_CORPSE) || (actor->flags6 & MF6_KILLED)) {
        strncpy(name, "Dead", VIZ_MAX_NAME_LEN);
        strncpy(name + 4, actor->GetClass()->TypeName.GetChars(), VIZ_MAX_NAME_LEN - 5);
    } else strncpy(name, actor->GetClass()->TypeName.GetChars(), VIZ_MAX_NAME_LEN - 1);
    name[VIZ_MAX_NAME_LEN - 1] = '\0';
}

inline unsigned int VIZ_GetActorId(AActor* actor){
    if(actor->viz_id == -1) {
        actor->viz_id = vizUniqueObjectsCount++;
    }
    return actor->viz_id;
}


/* Main functions */
/*--------------------------------------------------------------------------------------------------------------------*/

void VIZ_GameStateInit(){

    try {
        VIZSMRegion* gameStateRegion = &VIZ_SM_GAMESTATE;
        VIZ_SMCreateRegion(gameStateRegion, false, 0, sizeof(VIZGameState));
        vizGameStateSM = static_cast<VIZGameState *>(gameStateRegion->address);

        VIZ_DebugMsg(1, VIZ_FUNC, "gameStateOffset: %zu, gameStateSize: %zu", gameStateRegion->offset,
                     sizeof(VIZGameState));
    }
    catch(bip::interprocess_exception &ex){
        VIZ_Error(VIZ_FUNC, "Failed to create game state.");
    }

    vizGameStateSM->VERSION = VIZ_VERSION;
    strncpy(vizGameStateSM->VERSION_STR, VIZ_VERSION_STR, 16);
    vizGameStateSM->SM_SIZE = vizSMSize;

    vizGameStateSM->GAME_TIC = 0;
    vizGameStateSM->PLAYER_HEALTH = 0;
    vizGameStateSM->PLAYER_HAS_ACTOR = false;
    vizGameStateSM->PLAYER_DEAD = true;

    for(int i = 0; i < 9; ++i){
        vizGameStateSM->PLAYER_MOVEMENT[i] = 0;
    }
}

void VIZ_GameStateSMUpdate(){
    if(!vizGameStateSM) return;

    vizGameStateSM->SM_SIZE = vizSMSize;

    vizGameStateSM->SCREEN_WIDTH = vizScreenWidth;
    vizGameStateSM->SCREEN_HEIGHT = vizScreenHeight;
    vizGameStateSM->SCREEN_PITCH = vizScreenPitch;
    vizGameStateSM->SCREEN_SIZE = vizScreenSize;
    vizGameStateSM->SCREEN_FORMAT = *viz_screen_format;

    vizGameStateSM->DEPTH_BUFFER = *viz_depth && vizDepthMap;
    vizGameStateSM->LABELS = *viz_labels && vizLabels;
    vizGameStateSM->AUTOMAP = *viz_automap;
    vizGameStateSM->OBJECTS = *viz_objects;
    vizGameStateSM->SECTORS = *viz_sectors;
    vizGameStateSM->OPENAL_SOUND = ((stricmp(snd_backend, "openal") == 0) && GSnd && GSnd->IsValid());
    vizGameStateSM->AUDIO_BUFFER = *viz_soft_audio && vizGameStateSM->OPENAL_SOUND;
    vizGameStateSM->NOTIFICATIONS = *viz_notifications;

    for(int i = 0; i < VIZ_SM_REGION_COUNT; ++i){
        vizGameStateSM->SM_REGION_OFFSET[i] = vizSMRegion[i].offset;
        vizGameStateSM->SM_REGION_SIZE[i] = vizSMRegion[i].size;
        vizGameStateSM->SM_REGION_WRITEABLE[i] = vizSMRegion[i].writeable;
    }
}

void VIZ_GameStateTic(){
    if(!vizGameStateSM) return;

    VIZ_DebugMsg(2, VIZ_FUNC, "netgame: %d, multiplayer: %d, recording: %d, playback: %d, in_level: %d, map_tic: %d, map_ticlimit: %d",
                    netgame, multiplayer, demorecording, demoplayback, gamestate == GS_LEVEL, level.maptime, timelimit * TICRATE * 60);

    vizGameStateSM->GAME_TIC = (unsigned int)gametic;
    vizGameStateSM->GAME_STATE = gamestate;
    vizGameStateSM->GAME_ACTION = gameaction;
    vizGameStateSM->GAME_STATIC_SEED = staticrngseed;
    vizGameStateSM->GAME_SETTINGS_CONTROLLER = VIZ_PLAYER.settings_controller;
    vizGameStateSM->GAME_NETGAME = netgame;
    vizGameStateSM->GAME_MULTIPLAYER = multiplayer;
    vizGameStateSM->GAME_DEATHMATCH = (bool)deathmatch;
    vizGameStateSM->DEMO_RECORDING = demorecording;
    vizGameStateSM->DEMO_PLAYBACK = demoplayback;

    vizGameStateSM->MAP_END = gamestate != GS_LEVEL;
    vizGameStateSM->MAP_START_TIC = (unsigned int)level.starttime;
    vizGameStateSM->MAP_TIC = (unsigned int)level.maptime;
    vizGameStateSM->MAP_TICLIMIT = (unsigned int)(timelimit * TICRATE * 60);

    bool prevDead = vizGameStateSM->PLAYER_DEAD;

    if(VIZ_PLAYER.mo != NULL) {
        vizGameStateSM->PLAYER_HAS_ACTOR = true;
        vizGameStateSM->PLAYER_DEAD = VIZ_PLAYER.playerstate == PST_DEAD || VIZ_PLAYER.mo->health <= 0;
    }
    else {
        vizGameStateSM->PLAYER_HAS_ACTOR = false;
        vizGameStateSM->PLAYER_DEAD = true;
    }

    if(vizGameStateSM->PLAYER_DEAD && !prevDead) ++vizGameStateSM->PLAYER_DEATHCOUNT;

    vizGameStateSM->PLAYER_READY_TO_RESPAWN = VIZ_PLAYER.playerstate == PST_REBORN;
}

void VIZ_GameStateUpdate(){
    if(!vizGameStateSM) return;

    VIZ_GameStateUpdateVariables();

    if (!*viz_nocheat && !vizGameStateSM->MAP_END) {
        if (*viz_labels) VIZ_GameStateUpdateLabels();
        if (*viz_objects) VIZ_GameStateUpdateObjects();
        if (*viz_sectors) VIZ_GameStateUpdateSectors();
    }
    if (*viz_notifications) VIZ_GameStateUpdateNotifications();
}

void VIZ_GameStateUpdateVariables(){

    // Reward and ACS vars
    for(int i = 0; i < VIZ_GV_USER_COUNT; ++i) vizGameStateSM->MAP_USER_VARS[i] = ACS_GlobalVars[i+1];
    vizGameStateSM->MAP_REWARD = ACS_GlobalVars[0];

    // Player position GameVariables
    if(VIZ_PLAYER.mo != NULL && !*viz_nocheat) {
        vizGameStateSM->PLAYER_MOVEMENT[0] = VIZ_FixedToDouble(VIZ_PLAYER.mo->__pos.x); //X()
        vizGameStateSM->PLAYER_MOVEMENT[1] = VIZ_FixedToDouble(VIZ_PLAYER.mo->__pos.y); //Y()
        vizGameStateSM->PLAYER_MOVEMENT[2] = VIZ_FixedToDouble(VIZ_PLAYER.mo->__pos.z); //Z()
        vizGameStateSM->PLAYER_MOVEMENT[3] = VIZ_AngleToDouble(VIZ_PLAYER.mo->angle);
        vizGameStateSM->PLAYER_MOVEMENT[4] = VIZ_PitchToDouble(VIZ_PLAYER.mo->pitch);
        vizGameStateSM->PLAYER_MOVEMENT[5] = VIZ_AngleToDouble(VIZ_PLAYER.mo->roll);
        vizGameStateSM->PLAYER_MOVEMENT[6] = VIZ_FixedToDouble(VIZ_PLAYER.viewz) - vizGameStateSM->PLAYER_MOVEMENT[2];
        vizGameStateSM->PLAYER_MOVEMENT[7] = VIZ_FixedToDouble(VIZ_PLAYER.mo->velx);
        vizGameStateSM->PLAYER_MOVEMENT[8] = VIZ_FixedToDouble(VIZ_PLAYER.mo->vely);
        vizGameStateSM->PLAYER_MOVEMENT[9] = VIZ_FixedToDouble(VIZ_PLAYER.mo->velz);
    }

    if(VIZ_PLAYER.camera != NULL && !*viz_nocheat) {
        vizGameStateSM->CAMERA[0] = VIZ_FixedToDouble(VIZ_PLAYER.camera->__pos.x); //X()
        vizGameStateSM->CAMERA[1] = VIZ_FixedToDouble(VIZ_PLAYER.camera->__pos.y); //Y()
        vizGameStateSM->CAMERA[2] = VIZ_FixedToDouble(VIZ_PLAYER.viewz); //VIZ_FixedToDouble(VIZ_PLAYER.camera->__pos.z); //Z()
        vizGameStateSM->CAMERA[3] = VIZ_AngleToDouble(VIZ_PLAYER.camera->angle);
        vizGameStateSM->CAMERA[4] = VIZ_PitchToDouble(VIZ_PLAYER.camera->pitch);
        vizGameStateSM->CAMERA[5] = VIZ_AngleToDouble(VIZ_PLAYER.camera->roll);
        vizGameStateSM->CAMERA[6] = VIZ_PLAYER.FOV;
    }

    strncpy(vizGameStateSM->PLAYER_NAME, VIZ_PLAYER.userinfo.GetName(), VIZ_MAX_PLAYER_NAME_LEN);

    // Other
    int killed_players = 0;
    for (int i = 0; i < VIZ_MAX_PLAYERS; ++i){
        if (i != VIZ_PLAYER_NUM) killed_players += VIZ_PLAYER.frags[i];
    }
    vizGameStateSM->MAP_KILLCOUNT = level.killed_monsters;
    vizGameStateSM->MAP_ITEMCOUNT = level.found_items;
    vizGameStateSM->MAP_SECRETCOUNT = level.found_secrets;

    vizGameStateSM->PLAYER_KILLCOUNT = VIZ_PLAYER.killcount + killed_players;
    vizGameStateSM->PLAYER_ITEMCOUNT = VIZ_PLAYER.itemcount;
    vizGameStateSM->PLAYER_SECRETCOUNT = VIZ_PLAYER.secretcount;
    vizGameStateSM->PLAYER_FRAGCOUNT = VIZ_PLAYER.fragcount;

    vizGameStateSM->PLAYER_ATTACK_READY = (VIZ_PLAYER.WeaponState & WF_WEAPONREADY) != 0;
    vizGameStateSM->PLAYER_ALTATTACK_READY = (VIZ_PLAYER.WeaponState & WF_WEAPONREADYALT) != 0;
    vizGameStateSM->PLAYER_ON_GROUND = VIZ_PLAYER.onground;

    if (vizGameStateSM->PLAYER_HAS_ACTOR) vizGameStateSM->PLAYER_HEALTH = VIZ_PLAYER.mo->health;
        //else vizGameStateSM->PLAYER_HEALTH = VIZ_PLAYER.health;
    else vizGameStateSM->PLAYER_HEALTH = 0;

    vizGameStateSM->PLAYER_ARMOR = VIZ_CheckItem(NAME_BasicArmor);

    vizGameStateSM->PLAYER_SELECTED_WEAPON_AMMO = VIZ_CheckSelectedWeaponAmmo();
    vizGameStateSM->PLAYER_SELECTED_WEAPON = VIZ_CheckSelectedWeapon();

    for (unsigned int i = 0; i < VIZ_GV_SLOTS_SIZE; ++i) {
        vizGameStateSM->PLAYER_AMMO[i] = VIZ_CheckSlotAmmo(i);
        vizGameStateSM->PLAYER_WEAPON[i] = VIZ_CheckSlotWeapons(i);
    }

    // Player logger
    vizGameStateSM->PLAYER_HITCOUNT = vizPlayerLogger[VIZ_PLAYER_NUM].hitCount;
    vizGameStateSM->PLAYER_HITS_TAKEN = vizPlayerLogger[VIZ_PLAYER_NUM].hitsTaken;
    vizGameStateSM->PLAYER_DAMAGECOUNT = vizPlayerLogger[VIZ_PLAYER_NUM].dmgCount;
    vizGameStateSM->PLAYER_DAMAGE_TAKEN = vizPlayerLogger[VIZ_PLAYER_NUM].dmgTaken;

    // Multiplayer
    vizGameStateSM->PLAYER_NUMBER = (unsigned int)VIZ_PLAYER_NUM;
    vizGameStateSM->PLAYER_COUNT = 1;

    if(netgame || multiplayer){
        vizGameStateSM->PLAYER_COUNT = 0;

        for (size_t i = 0; i < VIZ_MAX_PLAYERS; ++i){
            if(playeringame[i]){
                ++vizGameStateSM->PLAYER_COUNT;
                vizGameStateSM->PLAYER_N_IN_GAME[i] = true;
                strncpy(vizGameStateSM->PLAYER_N_NAME[i], players[i].userinfo.GetName(), MAXPLAYERNAME);
                vizGameStateSM->PLAYER_N_NAME[i][MAXPLAYERNAME] = '\0';
                vizGameStateSM->PLAYER_N_FRAGCOUNT[i] = players[i].fragcount;
                if(players[i].cmd.ucmd.buttons != 0)
                    vizGameStateSM->PLAYER_N_LAST_ACTION_TIC[i] = (unsigned int)gametic;
                if(level.maptime >= viz_afk_timeout * 35
                   && vizGameStateSM->PLAYER_N_LAST_ACTION_TIC[i] < (unsigned int)(level.maptime - *viz_afk_timeout * 35)
                   && !players[i].userinfo.GetSpectator())
                    vizGameStateSM->PLAYER_N_AFK[i] = true;
                else vizGameStateSM->PLAYER_N_AFK[i] = false;
                vizGameStateSM->PLAYER_N_LAST_KILL_TIC[i] = players[i].lastkilltime;
            }
            else{
                vizGameStateSM->PLAYER_N_IN_GAME[i] = false;
                vizGameStateSM->PLAYER_N_FRAGCOUNT[i] = 0;
                vizGameStateSM->PLAYER_N_AFK[i] = false;
                vizGameStateSM->PLAYER_N_LAST_ACTION_TIC[i] = 0;
                vizGameStateSM->PLAYER_N_LAST_KILL_TIC[i] = 0;
            }
        }
    }
}

void VIZ_GameStateUpdateLabels(){

    unsigned int labelCount = 0;
    if(vizLabels != NULL){

        VIZ_DebugMsg(4, VIZ_FUNC, "number of sprites: %d", gametic, vizLabels->getSprites().size());

        //TODO sort vizLabels->sprites

        for(auto& sprite : vizLabels->sprites){
            if(sprite.labeled && sprite.pointCount > 0){
                VIZLabel *vizLabel = &vizGameStateSM->LABEL[labelCount++];
                const PClass *actorClass;

                vizLabel->objectId = VIZ_GetActorId(sprite.actor);
                vizLabel->value = sprite.label;

                // Handle DehackedPickup
                if (sprite.actor->IsKindOf(RUNTIME_CLASS(ADehackedPickup))) {
                    actorClass = static_cast<ADehackedPickup *>(sprite.actor)->DetermineType();
                    strncpy(vizLabel->objectName, actorClass->TypeName.GetChars(), VIZ_MAX_NAME_LEN);
                } else {
                    actorClass = sprite.actor->GetClass();
                    VIZ_CopyActorName(sprite.actor, vizLabel->objectName);
                }

                // Check for special cases before matching category
                if (strncmp(vizLabel->objectName, "Dead", 4) == 0) {
                    // Align with the behavior of VIZ_CopyActorName
                    strncpy(vizLabel->objectCategory, "Gore", VIZ_MAX_NAME_LEN);
                } else if (VIZ_PLAYER.mo == sprite.actor) {
                    // Detect whether this object is current player
                    strncpy(vizLabel->objectCategory, "Self", VIZ_MAX_NAME_LEN);
                } else {
                    PClass *currentClass = const_cast<PClass *>(actorClass);
                    bool isUnknown = true;

                    while (currentClass != nullptr)
                    {
                        // Convert to lowercase for lookup since the mapping uses casefolded names
                        std::string className = currentClass->TypeName.GetChars();

                        // Special case: HealthPickup is parallel to Health but is Health category
                        if (className.compare("HealthPickup") == 0) {
                            className = "Health";
                        }

                        // Class could be a category (like Health for CustomMedikit and Poison)
                        if (std::find(categories.begin(), categories.end(), className) != categories.end()) {
                            strncpy(vizLabel->objectCategory, className.c_str(), VIZ_MAX_NAME_LEN);
                            isUnknown = false;
                            break;
                        }

                        std::transform(className.begin(), className.end(), className.begin(), tolower);

                        auto categoryIt = classToCategory.find(className);
                        if (categoryIt != classToCategory.end()) {
                            strncpy(vizLabel->objectCategory, categoryIt->second.c_str(), VIZ_MAX_NAME_LEN);
                            vizLabel->objectCategory[VIZ_MAX_NAME_LEN-1] = '\0';  // Safe-guard against long names
                            isUnknown = false;
                            break;
                        } else {
                            currentClass = currentClass->ParentClass;
                        }
                    }

                    if (isUnknown) {
                        strncpy(vizLabel->objectCategory, "Unknown", VIZ_MAX_NAME_LEN);
                    }
                }

                if(sprite.minX >= vizGameStateSM->SCREEN_WIDTH) sprite.minX = vizGameStateSM->SCREEN_WIDTH - 1;
                if(sprite.minY >= vizGameStateSM->SCREEN_HEIGHT) sprite.minY = vizGameStateSM->SCREEN_HEIGHT - 1;
                if(sprite.maxX >= vizGameStateSM->SCREEN_WIDTH) sprite.maxX = vizGameStateSM->SCREEN_WIDTH - 1;
                if(sprite.maxY >= vizGameStateSM->SCREEN_HEIGHT) sprite.maxY = vizGameStateSM->SCREEN_HEIGHT - 1;

                vizLabel->position[0] = sprite.minX;
                vizLabel->position[1] = sprite.minY;
                vizLabel->size[0] = sprite.maxX - sprite.minX;
                vizLabel->size[1] = sprite.maxY - sprite.minY;

                vizLabel->objectPosition[0] = VIZ_FixedToDouble(sprite.actor->__pos.x);
                vizLabel->objectPosition[1] = VIZ_FixedToDouble(sprite.actor->__pos.y);
                vizLabel->objectPosition[2] = VIZ_FixedToDouble(sprite.actor->__pos.z);
                vizLabel->objectPosition[3] = VIZ_AngleToDouble(sprite.actor->angle);
                vizLabel->objectPosition[4] = VIZ_PitchToDouble(sprite.actor->pitch);
                vizLabel->objectPosition[5] = VIZ_AngleToDouble(sprite.actor->roll);
                vizLabel->objectPosition[6] = VIZ_FixedToDouble(sprite.actor->velx);
                vizLabel->objectPosition[7] = VIZ_FixedToDouble(sprite.actor->vely);
                vizLabel->objectPosition[8] = VIZ_FixedToDouble(sprite.actor->velz);

                VIZ_DebugMsg(4, VIZ_FUNC, "labelCount: %d, objectId: %d, objectName: %s, value %d",
                                labelCount, vizLabel->objectId, vizLabel->objectName, vizLabel->value);
            }
            if(labelCount >= VIZ_MAX_LABELS) break;
        }
    }
    vizGameStateSM->LABEL_COUNT = labelCount;
}

void VIZ_GameStateUpdateObjects(){
    unsigned int objectCount = 0;

    // Iterate over sectors
    for (int i = 0; i < numsectors; ++i) {
        sector_t *sector = &sectors[i];

        // Handle all things in sector
        for (AActor *actor = sector->thinglist; actor != NULL; actor = actor->snext) {
            VIZObject *vizObject = &vizGameStateSM->OBJECT[objectCount++];

            vizObject->id = VIZ_GetActorId(actor);
            VIZ_CopyActorName(actor, vizObject->name);
            vizObject->position[0] = VIZ_FixedToDouble(actor->__pos.x);
            vizObject->position[1] = VIZ_FixedToDouble(actor->__pos.y);
            vizObject->position[2] = VIZ_FixedToDouble(actor->__pos.z);
            vizObject->position[3] = VIZ_AngleToDouble(actor->angle);
            vizObject->position[4] = VIZ_PitchToDouble(actor->pitch);
            vizObject->position[5] = VIZ_AngleToDouble(actor->roll);
            vizObject->position[6] = VIZ_FixedToDouble(actor->velx);
            vizObject->position[7] = VIZ_FixedToDouble(actor->vely);
            vizObject->position[8] = VIZ_FixedToDouble(actor->velz);

            VIZ_DebugMsg(4, VIZ_FUNC, "objectCount: %d, id: %d, name: %s", objectCount, vizObject->id, vizObject->name);
            if(objectCount >= VIZ_MAX_OBJECTS) break;
        }
    }

    vizGameStateSM->OBJECT_COUNT = objectCount;
}

void VIZ_GameStateUpdateSectors(){
    //std::unordered_map<sector_t *, int> sectorIds(numsectors);
    std::unordered_map<line_t *, int> lineIds(numlines);

    unsigned int lineCount = 0;
    for(int i = 0; i < numlines; ++i){
        line_t *line = &lines[i];
        VIZLine *vizLine = &vizGameStateSM->LINE[lineCount++];

        vizLine->position[0] = VIZ_FixedToDouble(line->v1->x);
        vizLine->position[1] = VIZ_FixedToDouble(line->v1->y);
        vizLine->position[2] = VIZ_FixedToDouble(line->v2->x);
        vizLine->position[3] = VIZ_FixedToDouble(line->v2->y);

        //vizLine->frontSector = sectorIds[line->frontsector];
        //vizLine->backSector = sectorIds[line->backsector];

        lineIds.insert({line, i});
        vizLine->isBlocking = (line->flags & (ML_BLOCKING|ML_BLOCKEVERYTHING|ML_BLOCK_PLAYERS));

        VIZ_DebugMsg(4, VIZ_FUNC, "line: %d, position: (%f, %f), (%f, %f), isBlocking: %d",
                i, vizLine->position[0], vizLine->position[1], vizLine->position[2], vizLine->position[3], vizLine->isBlocking);

        if(lineCount >= VIZ_MAX_LINES) break;
    }

    VIZ_DebugMsg(4, VIZ_FUNC, "lineCount: %d, numlines: %d", lineCount, numlines);

    vizGameStateSM->LINE_COUNT = lineCount;
    assert(lineCount == numlines);

    unsigned int sectorCount = 0;
    for(int i = 0; i < numsectors; ++i){
        sector_t *sector = &sectors[i];
        VIZSector *vizSector = &vizGameStateSM->SECTOR[sectorCount++];

        vizSector->ceilingHeight = VIZ_FixedToDouble(sector->ceilingplane.d);
        vizSector->floorHeight = VIZ_FixedToDouble(sector->floorplane.d);

        unsigned int sectorLineCount = 0;
        for(int l = 0; l < sector->linecount; ++l){
            line_t *line = sector->lines[l];
            vizSector->lines[sectorLineCount++] = lineIds[line];
            //auto lineId = lineIds.find(line);
            //if(lineId != lineIds.end()) vizSector->lines[lineCount++] = lineId->second;
            //else  = lineIds.insert({line, lineIds.size()}).second;
        }
        vizSector->lineCount = sectorLineCount;
        assert(sectorLineCount == sector->linecount);

        //sectorIds.insert({sector, sectorIds.size()});
        if(sectorCount >= VIZ_MAX_SECTORS) break;
    }

    VIZ_DebugMsg(4, VIZ_FUNC, "sectorCount: %d, numsectors: %d", sectorCount, numsectors);

    vizGameStateSM->SECTOR_COUNT = sectorCount;
    assert(sectorCount == numsectors);
}

void VIZ_GameStateUpdateNotifications(){
    if(!vizGameStateSM) return;

    int bufferStartTic = gametic - *viz_notifications_tics;

    VIZ_DebugMsg(4, VIZ_FUNC, "gametic: %d, bufferStartTic: %d, number of tics in notifications: %d", gametic, bufferStartTic, vizNotifications.size());

    // Update the notifications buffer
    // Remove old notifications from the list
    std::string newNotificationsBuffer = "";
    int notificationCount = 0;
    if (!vizNotifications.empty()) {
        int toRemove = 0;
        for (const auto& notification : vizNotifications) {
            if (notification.gametic >= bufferStartTic && notification.gametic >= level.starttime) {
                for (const auto& message : notification.messages) {
                    newNotificationsBuffer += message;
                    if (newNotificationsBuffer.back() != '\n') newNotificationsBuffer += '\n';
                    ++notificationCount;
                }
            } else ++toRemove;
        }
        if (toRemove > 0) vizNotifications.erase(vizNotifications.begin(), vizNotifications.begin() + toRemove);
    }

    VIZ_DebugMsg(4, VIZ_FUNC, "notifications buffer length: %d, notification count: %d, notifications: %s", newNotificationsBuffer.length(), notificationCount, newNotificationsBuffer.c_str());

    // Copy the notifications buffer to shared memory, ensuring it's truncated if too long.
    if (newNotificationsBuffer.length() >= VIZ_MAX_NOTIFICATIONS_CHARS) {
        // Truncate and add ellipsis
        std::string truncated = newNotificationsBuffer.substr(0, VIZ_MAX_NOTIFICATIONS_CHARS - 4);
        truncated += "...";
        strncpy(vizGameStateSM->NOTIFICATIONS_TEXT, truncated.c_str(), VIZ_MAX_NOTIFICATIONS_CHARS - 1);
        vizGameStateSM->NOTIFICATIONS_TEXT_SIZE = truncated.length();
    } else {
        strncpy(vizGameStateSM->NOTIFICATIONS_TEXT, newNotificationsBuffer.c_str(), VIZ_MAX_NOTIFICATIONS_CHARS - 1);
        vizGameStateSM->NOTIFICATIONS_TEXT_SIZE = newNotificationsBuffer.length();
    }
    vizGameStateSM->NOTIFICATIONS_TEXT[VIZ_MAX_NOTIFICATIONS_CHARS - 1] = '\0';
}

void VIZ_GameStateInitNew(){
    if(!vizGameStateSM) return;

    if(*viz_loop_map && !level.MapName.Compare(level.NextMap)){
        level.NextMap = level.MapName;
        level.NextSecretMap = level.MapName;
    }

    for (size_t i = 0; i < VIZ_MAX_PLAYERS; ++i) {
        vizPlayerLogger[i].reset();
    }

    vizUniqueObjectsCount = 0;
}

void VIZ_GameStateClose(){
    VIZ_SMDeleteRegion(&VIZ_SM_GAMESTATE);
}

void VIZ_PrintPlayers(){
    printf("players state: tic %d: player_count: %d, players:\n", gametic, vizGameStateSM->PLAYER_COUNT);
    for (size_t i = 0; i < VIZ_MAX_PLAYERS; ++i) {
        if(playeringame[i]){
            APlayerPawn* player = players[i].mo;
            printf("no: %lu, name: %s, pos: %f %f %f, rot: %f %f %f, vel: %f %f %f\n", i + 1, players[i].userinfo.GetName(),
                   VIZ_FixedToDouble(player->__pos.x), VIZ_FixedToDouble(player->__pos.y), VIZ_FixedToDouble(player->__pos.z),
                   VIZ_AngleToDouble(player->angle), VIZ_PitchToDouble(player->pitch), VIZ_AngleToDouble(player->roll),
                   VIZ_FixedToDouble(player->velx), VIZ_FixedToDouble(player->vely), VIZ_FixedToDouble(player->velz));
            printf("no: %lu, name: %s, dmgCount: %d, hitCount: %d\n", i + 1, players[i].userinfo.GetName(),
                   vizPlayerLogger[i].dmgCount, vizPlayerLogger[i].hitCount);
        }
    }
}
