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

#ifndef __VIZDOOM_CONTROLLER_H__
#define __VIZDOOM_CONTROLLER_H__

#include "ViZDoomDefines.h"

#include <boost/asio.hpp>
#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/thread.hpp>
#include <string>
#include <vector>

namespace vizdoom{

    namespace b         = boost;
    namespace bip       = boost::interprocess;
    namespace ba        = boost::asio;


/* Shared memory's settings */
#define SM_NAME_BASE        "ViZDoomSM"

/* Message queues' settings */
#define MQ_NAME_CTR_BASE    "ViZDoomMQCtr"
#define MQ_NAME_DOOM_BASE   "ViZDoomMQDoom"
#define MQ_MAX_MSG_NUM      64
#define MQ_MAX_MSG_SIZE     sizeof(DoomController::MessageCommandStruct)
#define MQ_MAX_CMD_LEN      128

/* Messages' codes */
#define MSG_CODE_DOOM_DONE              11
#define MSG_CODE_DOOM_CLOSE             12
#define MSG_CODE_DOOM_ERROR             13
#define MSG_CODE_DOOM_PROCESS_EXIT      14

#define MSG_CODE_TIC                    21
#define MSG_CODE_UPDATE                 22
#define MSG_CODE_TIC_N_UPDATE           23
#define MSG_CODE_COMMAND                24
#define MSG_CODE_CLOSE                  25
#define MSG_CODE_ERROR                  26

#define MSG_CODE_SIGNAL_INT_ABRT_TERM   30

/* OSes */
#ifdef __linux__
    #define OS_LINUX
#elif _WIN32
    #define OS_WIN
#elif __APPLE__
    #define OS_OSX
#endif

    class DoomController {

    public:

        /* SM structs */
        /*------------------------------------------------------------------------------------------------------------*/

        struct InputStruct {
            int BT[ButtonsNumber];
            bool BT_AVAILABLE[ButtonsNumber];
            int BT_MAX_VALUE[DeltaButtonsNumber];
        };

        struct GameVariablesStruct {
            unsigned int VERSION;
            char VERSION_STR[8];

            unsigned int GAME_TIC;
            int GAME_STATE;
            int GAME_ACTION;
            unsigned int GAME_SEED;
            unsigned int GAME_RNG_SEED;
            bool GAME_SETTINGS_CONTROLLER;
            bool NET_GAME;

            unsigned int SCREEN_WIDTH;
            unsigned int SCREEN_HEIGHT;
            size_t SCREEN_PITCH;
            size_t SCREEN_SIZE;
            int SCREEN_FORMAT;

            unsigned int MAP_START_TIC;
            unsigned int MAP_TIC;

            int MAP_REWARD;

            int MAP_USER_VARS[UserVariablesNumber];

            int MAP_KILLCOUNT;
            int MAP_ITEMCOUNT;
            int MAP_SECRETCOUNT;
            bool MAP_END;

            bool PLAYER_HAS_ACTOR;
            bool PLAYER_DEAD;

            char PLAYER_NAME[16];
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

            int PLAYER_AMMO[SlotsNumber];
            int PLAYER_WEAPON[SlotsNumber];

            int PLAYERS_COUNT;
            char PLAYERS_NAME[MaxNumberOfPlayers][16];
            int PLAYERS_FRAGCOUNT[MaxNumberOfPlayers];
        };

        DoomController();
        ~DoomController();

        /* Flow control */
        /*------------------------------------------------------------------------------------------------------------*/

        bool init();
        void close();
        void restart();

        void intSignal();   //must be public

        bool isTicPossible();
        void tic();
        void tic(bool update);
        void tics(unsigned int tics);
        void tics(unsigned int tics, bool update);
        void restartMap();
        void respawnPlayer();
        bool isDoomRunning();
        void sendCommand(std::string command);

        /* General game settings */
        /*------------------------------------------------------------------------------------------------------------*/

        unsigned int getSeed();
        unsigned int getRngSeed();
        void setRngSeed(unsigned int seed);
        void clearRngSeed();
        void setUseRngSeed(bool use);
        bool isUseRngSeed();

        std::string getInstanceId();
        void setInstanceId(std::string id);

        std::string getExePath();
        void setExePath(std::string path);

        std::string getIwadPath();
        void setIwadPath(std::string path);

        std::string getFilePath();
        void setFilePath(std::string path);

        std::string getMap();
        void setMap(std::string map);

        int getSkill();
        void setSkill(int skill);

        std::string getConfigPath();
        void setConfigPath(std::string path);

        unsigned int getMapStartTime();
        void setMapStartTime(unsigned int tics);

        unsigned int getMapTimeout();
        void setMapTimeout(unsigned int tics);

        bool isMapLastTic();
        bool isMapFirstTic();
        bool isMapEnded();
        unsigned int getMapLastTic();

        void setNoConsole(bool console);
        void setNoSound(bool noSound);

        void addCustomArg(std::string arg);
        void clearCustomArgs();


        /* Rendering getters and setters */
        /*------------------------------------------------------------------------------------------------------------*/

        void setWindowHidden(bool windowHidden);
        void setNoXServer(bool noXServer);

        void setRenderHud(bool hud);
        void setRenderWeapon(bool weapon);
        void setRenderCrosshair(bool crosshair);
        void setRenderDecals(bool decals);
        void setRenderParticles(bool particles);

        void setScreenResolution(unsigned int width, unsigned int height);
        void setScreenWidth(unsigned int width);
        void setScreenHeight(unsigned int height);
        void setScreenFormat(ScreenFormat format);

        ScreenFormat getScreenFormat();
        unsigned int getScreenWidth();
        unsigned int getScreenHeight();
        unsigned int getScreenChannels();
        unsigned int getScreenDepth();
        size_t getScreenPitch();
        size_t getScreenSize();

        uint8_t * const getScreen();


        /* Buttons getters and setters */
        /*------------------------------------------------------------------------------------------------------------*/

        InputStruct * const getInput();
        GameVariablesStruct * const getGameVariables();

        int getButtonState(Button button);
        void setButtonState(Button button, int state);
        void toggleButtonState(Button button);
        bool isButtonAvailable(Button button);
        void setButtonAvailable(Button button, bool set);
        void resetButtons();
        void disableAllButtons();
        void setButtonMaxValue(Button button, int value);
        int getButtonMaxValue(Button button);
        void availableAllButtons();

        bool isAllowDoomInput();
        void setAllowDoomInput(bool set);

        bool isRunDoomAsync();
        void setRunDoomAsync(bool set);


        /* GameVariables getters */
        /*------------------------------------------------------------------------------------------------------------*/

        int getGameVariable(GameVariable var);

        int getGameTic();
        bool isNetGame();

        int getMapTic();
        int getMapReward();

        int getMapKillCount();
        int getMapItemCount();
        int getMapSecretCount();

        bool isPlayerDead();

        int getPlayerKillCount();
        int getPlayerItemCount();
        int getPlayerSecretCount();
        int getPlayerFragCount();
        int getPlayerDeathCount();

        int getPlayerHealth();
        int getPlayerArmor();
        bool isPlayerOnGround();
        bool isPlayerAttackReady();
        bool isPlayerAltAttackReady();
        int getPlayerSelectedWeaponAmmo();
        int getPlayerSelectedWeapon();

        int getPlayerAmmo(unsigned int slot);
        int getPlayerWeapon(unsigned int slot);
        int getUser(unsigned int slot);

    private:

        /* Flow */
        /*------------------------------------------------------------------------------------------------------------*/

        bool doomRunning;
        bool doomWorking;

        void waitForDoomStart();
        void waitForDoomWork();
        void waitForDoomMapStartTime();
        void createDoomArgs();
        void launchDoom();
        void handleSignals();


        /* Seed */
        /*------------------------------------------------------------------------------------------------------------*/

        void generateInstanceId();

        bool useRngSeed;
        int rngSeed;
        std::string instanceId;


        /* Threads */
        /*------------------------------------------------------------------------------------------------------------*/

        b::thread *doomThread;
        ba::io_service ioService;
        b::thread *signalThread;
        //bpr::child doomProcess;


        /* Message queues */
        /*------------------------------------------------------------------------------------------------------------*/

        struct MessageSignalStruct {
            uint8_t code;
        };

        struct MessageCommandStruct {
            uint8_t code;
            char command[MQ_MAX_CMD_LEN];
        };

        void MQInit();
        void MQControllerSend(uint8_t code);
        void MQDoomSend(uint8_t code);
        void MQDoomSend(uint8_t code, const char *command);
        void MQControllerRecv(void *msg, size_t &size, unsigned int &priority);
        void MQClose();

        bip::message_queue *MQController;
        bip::message_queue *MQDoom;
        std::string MQControllerName;
        std::string MQDoomName;


        /* Shared memory */
        /*------------------------------------------------------------------------------------------------------------*/

        void SMInit();
        void SMClose();

        bip::shared_memory_object SM;
        std::string SMName;

        bip::mapped_region *InputSMRegion;
        InputStruct *input;
        InputStruct *_input;

        bip::mapped_region *GameVariablesSMRegion;
        GameVariablesStruct *gameVariables;

        bip::mapped_region *ScreenSMRegion;
        uint8_t *screen;


        /* Settings */
        /*------------------------------------------------------------------------------------------------------------*/

        unsigned int screenWidth, screenHeight, screenChannels, screenDepth;
        size_t screenPitch, screenSize;
        ScreenFormat screenFormat;

        bool hud, weapon, crosshair, decals, particles;

        bool windowHidden, noXServer;

        bool noConsole;
        bool noSound;

        std::string exePath;
        std::string iwadPath;
        std::string filePath;
        std::string map;
        std::string configPath;
        int skill;

        bool allowDoomInput;
        bool runDoomAsync;

        unsigned int mapStartTime;
        unsigned int mapTimeout;
        unsigned int mapRestartCount;
        bool mapRestarting;
        unsigned int mapLastTic;

        std::vector <std::string> customArgs;
        std::vector <std::string> doomArgs;

    };

}

#endif
