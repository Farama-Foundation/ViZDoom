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

#ifndef __VIZDOOM_SHAREDMEMORY_H__
#define __VIZDOOM_SHAREDMEMORY_H__

#include "ViZDoomConsts.h"
#include "ViZDoomTypes.h"

#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <cstdint>

#define SM_REGION_COUNT 7

#define MAX_LABELS 256
#define MAX_OBJECTS 4096
#define MAX_SECTORS 2048
#define MAX_LINES 8192

namespace vizdoom {

    namespace b         = boost;
    namespace bip       = boost::interprocess;

    /* SM region */
    /*----------------------------------------------------------------------------------------------------------------*/

    struct SMRegion {
        bip::mapped_region *region;
        void *address;
        size_t offset;
        size_t size;
        bool writeable;

        SMRegion() {
            this->region = nullptr;
            this->address = nullptr;
            this->offset = 0;
            this->size = 0;
            this->writeable = false;
        };
    };

    /* SM structs */
    /*----------------------------------------------------------------------------------------------------------------*/

    struct SMLabel {
        uint8_t value;
        unsigned int position[2];
        unsigned int size[2];

        unsigned int objectId;
        double objectPosition[9];
        char objectName[MAX_NAME_LENGTH];
    };

    struct SMObject {
        unsigned int id;
        double position[9];
        char name[MAX_NAME_LENGTH];
    };

    struct SMSector{
        double floorHeight;
        double ceilingHeight;
        unsigned int lineCount;
        unsigned int lines[128];
    };

    struct SMLine{
        double position[4];
        bool isBlocking;
    };

    struct SMGameState {
        // VERSION
        unsigned int VERSION;
        char VERSION_STR[8];

        // SM
        size_t SM_SIZE;
        size_t SM_REGION_OFFSET[SM_REGION_COUNT];
        size_t SM_REGION_SIZE[SM_REGION_COUNT];
        bool SM_REGION_WRITEABLE[SM_REGION_COUNT];

        // GAME
        unsigned int GAME_TIC;
        int GAME_STATE;
        int GAME_ACTION;
        unsigned int GAME_STATIC_SEED;
        bool GAME_SETTINGS_CONTROLLER;
        bool GAME_NETGAME;
        bool GAME_MULTIPLAYER;
        bool GAME_DEATHMATCH;
        bool DEMO_RECORDING;
        bool DEMO_PLAYBACK;

        // SCREEN
        unsigned int SCREEN_WIDTH;
        unsigned int SCREEN_HEIGHT;
        size_t SCREEN_PITCH;
        size_t SCREEN_SIZE;
        int SCREEN_FORMAT;

        bool DEPTH_BUFFER;
        bool LABELS;
        bool AUTOMAP;
        bool OBJECTS;
        bool SECTORS;

        // MAP
        unsigned int MAP_START_TIC;
        unsigned int MAP_TIC;
        unsigned int MAP_TICLIMIT;

        int MAP_REWARD;
        int MAP_USER_VARS[USER_VARIABLE_COUNT];

        int MAP_KILLCOUNT;
        int MAP_ITEMCOUNT;
        int MAP_SECRETCOUNT;
        bool MAP_END;

        // PLAYER
        bool PLAYER_HAS_ACTOR;
        bool PLAYER_DEAD;

        char PLAYER_NAME[MAX_PLAYER_NAME_LENGTH];
        int PLAYER_KILLCOUNT;
        int PLAYER_ITEMCOUNT;
        int PLAYER_SECRETCOUNT;
        int PLAYER_FRAGCOUNT;
        int PLAYER_DEATHCOUNT;

        bool PLAYER_ON_GROUND;

        int PLAYER_HEALTH;
        int PLAYER_ARMOR;

        bool PLAYER_ATTACK_READY;
        bool PLAYER_ALTATTACK_READY;

        int PLAYER_SELECTED_WEAPON;
        int PLAYER_SELECTED_WEAPON_AMMO;

        int PLAYER_AMMO[SLOT_COUNT];
        int PLAYER_WEAPON[SLOT_COUNT];

        double PLAYER_MOVEMENT[10];
        double CAMERA[7];

        bool PLAYER_READY_TO_RESPAWN;
        unsigned int PLAYER_NUMBER;

        // OTHER PLAYERS
        unsigned int PLAYER_COUNT;
        bool PLAYER_N_IN_GAME[MAX_PLAYERS];
        char PLAYER_N_NAME[MAX_PLAYERS][MAX_PLAYER_NAME_LENGTH];
        int PLAYER_N_FRAGCOUNT[MAX_PLAYERS];
        bool PLAYER_N_AFK[MAX_PLAYERS];
        unsigned int PLAYER_N_LAST_ACTION_TIC[MAX_PLAYERS];
        unsigned int PLAYER_N_LAST_KILL_TIC[MAX_PLAYERS];

        // LABELS
        unsigned int LABEL_COUNT;
        SMLabel LABEL[MAX_LABELS];

        // OBJECTS
        unsigned int OBJECT_COUNT;
        SMObject OBJECT[MAX_OBJECTS];

        // SECTORS
        unsigned int SECTOR_COUNT;
        SMSector SECTOR[MAX_SECTORS];

        // LINES
        unsigned int LINE_COUNT;
        SMLine LINE[MAX_LINES];

        // LOGGED
        int PLAYER_HITCOUNT;
        int PLAYER_HITS_TAKEN;
        int PLAYER_DAMAGECOUNT;
        int PLAYER_DAMAGE_TAKEN;
    };

    struct SMInputState {
        double BT[BUTTON_COUNT];
        bool BT_AVAILABLE[BUTTON_COUNT];
        double BT_MAX_VALUE[DELTA_BUTTON_COUNT];
        double CMD_BT[BUTTON_COUNT];
    };

    /* SM class */
    /*----------------------------------------------------------------------------------------------------------------*/

    class SharedMemory {

    public:
        SharedMemory(std::string name);
        ~SharedMemory();

        void init();
        void update();
        void close();

        SMGameState *getGameState();
        SMInputState *getInputState();
        uint8_t *getScreenBuffer();
        uint8_t *getDepthBuffer();
        uint8_t *getLabelsBuffer();
        uint8_t *getAutomapBuffer();
        int16_t *getAudioBuffer();

    private:
        bip::shared_memory_object sm;
        bip::offset_t size;
        std::string name;

        void mapRegion(SMRegion *regionPtr);

        void deleteRegion(SMRegion *regionPtr);

        //0 - GameState, 1 - InputState, 2 - ScreenBuffer, 3 - DepthBuffer, 4 - LabelsBuffer, 5 - AutomapBuffer, 6 - AudioBuffer
        SMRegion region[7];
    };
}

#endif
