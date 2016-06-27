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
#include <boost/random.hpp>
#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/thread.hpp>
#include <string>
#include <vector>

namespace vizdoom{

    namespace b         = boost;
    namespace ba        = boost::asio;
    namespace bip       = boost::interprocess;
    namespace br        = boost::random;
    namespace bs        = boost::system;

#define INSTANCE_ID_LENGHT 10

/* Shared memory's settings */
#define SM_NAME_BASE        "ViZDoomSM"

/* Message queues' settings */
#define MQ_NAME_CTR_BASE    "ViZDoomMQCtr"
#define MQ_NAME_DOOM_BASE   "ViZDoomMQDoom"
#define MQ_MAX_MSG_NUM      64
#define MQ_MAX_MSG_SIZE     sizeof(DoomController::Message)
#define MQ_MAX_CMD_LEN      128

/* Messages' codes */
#define MSG_CODE_DOOM_DONE              11
#define MSG_CODE_DOOM_CLOSE             12
#define MSG_CODE_DOOM_ERROR             13
#define MSG_CODE_DOOM_PROCESS_EXIT      14

#define MSG_CODE_TIC                    21
#define MSG_CODE_UPDATE                 22
#define MSG_CODE_TIC_AND_UPDATE         23
#define MSG_CODE_COMMAND                24
#define MSG_CODE_CLOSE                  25
#define MSG_CODE_ERROR                  26

#define MSG_CODE_SIG                    30
#define MSG_CODE_SIGINT                 30 + SIGINT
#define MSG_CODE_SIGABRT                30 + SIGABRT
#define MSG_CODE_SIGTERM                30 + SIGTERM

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

        struct GameState {
            unsigned int VERSION;
            char VERSION_STR[8];
            size_t SM_SIZE;

            unsigned int GAME_TIC;
            int GAME_STATE;
            int GAME_ACTION;
            unsigned int GAME_STATIC_SEED;
            bool GAME_SETTINGS_CONTROLLER;
            bool GAME_NETGAME;
            bool GAME_MULTIPLAYER;
            bool DEMO_RECORDING;
            bool DEMO_PLAYBACK;

            // SCREEN
            unsigned int SCREEN_WIDTH;
            unsigned int SCREEN_HEIGHT;
            size_t SCREEN_PITCH;
            size_t SCREEN_SIZE;
            int SCREEN_FORMAT;

            // MAP
            unsigned int MAP_START_TIC;
            unsigned int MAP_TIC;

            int MAP_REWARD;
            int MAP_USER_VARS[UserVariableCount];

            int MAP_KILLCOUNT;
            int MAP_ITEMCOUNT;
            int MAP_SECRETCOUNT;
            bool MAP_END;

            // PLAYER
            bool PLAYER_HAS_ACTOR;
            bool PLAYER_DEAD;

            char PLAYER_NAME[MaxPlayerNameLength];
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

            int PLAYER_AMMO[SlotCount];
            int PLAYER_WEAPON[SlotCount];

            bool PLAYER_READY_TO_RESPAWN;
            unsigned int PLAYER_NUMBER;

            // OTHER PLAYERS
            unsigned int PLAYER_COUNT;
            bool PLAYERS_IN_GAME[MaxPlayers];
            char PLAYERS_NAME[MaxPlayers][MaxPlayerNameLength];
            int PLAYERS_FRAGCOUNT[MaxPlayers];
        };

        struct InputState {
            int BT[ButtonCount];
            bool BT_AVAILABLE[ButtonCount];
            int BT_MAX_VALUE[DeltaButtonCount];
        };

        DoomController();
        ~DoomController();


        /* Flow control */
        /*------------------------------------------------------------------------------------------------------------*/

        bool init();
        void close();
        void restart();

        bool isTicPossible();
        void tic(bool update = true);
        void tics(unsigned int tics, bool update = true);
        void restartMap();
        void respawnPlayer();
        bool isDoomRunning();
        void sendCommand(std::string command);

        void setTicrate(unsigned int ticrate);
        unsigned int getTicrate();

        unsigned int getDoomRngSeed();
        void setDoomRngSeed(unsigned int seed);
        void clearDoomRngSeed();

        unsigned int getInstanceRngSeed();
        void setInstanceRngSeed(unsigned int seed);

        std::string getMap();
        void setMap(std::string map, std::string demoPath = "");
        void playDemo(std::string demoPath);


        /* General game settings */
        /*------------------------------------------------------------------------------------------------------------*/

        std::string getExePath();
        void setExePath(std::string exePath);

        std::string getIwadPath();
        void setIwadPath(std::string iwadPath);

        std::string getFilePath();
        void setFilePath(std::string filePath);

        int getSkill();
        void setSkill(int skill);

        std::string getConfigPath();
        void setConfigPath(std::string configPath);

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

        void setDepthBufferEnabled(bool depthBuffer);
        bool isDepthBufferEnabled();

        void setLevelMapEnabled(bool map);
        //void setLevelMapMode(MapMode mode);
        bool isLevelMapEnabled();

        void setLabelsEnabled(bool labels);
        bool isLabelsEnabled();

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

        InputState * const getInput();
        GameState * const getGameState();

        int getButtonState(Button button);
        void setButtonState(Button button, int state);
        void toggleButtonState(Button button);
        bool isButtonAvailable(Button button);
        void setButtonAvailable(Button button, bool set);
        void resetButtons();
        void disableAllButtons();
        void setButtonMaxValue(Button button, unsigned int value);
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
        bool isMultiplayerGame();
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
        bool doomRecordingMap;

        void waitForDoomStart();
        void waitForDoomWork();
        void waitForDoomMapStartTime();
        void createDoomArgs();
        void launchDoom();

        /* Seed */
        /*------------------------------------------------------------------------------------------------------------*/

        void generateInstanceId();

        bool seedDoomRng;
        unsigned int doomRngSeed;
        unsigned int instanceRngSeed;

        br::mt19937 instanceRng;
        std::string instanceId;


        /* Threads */
        /*------------------------------------------------------------------------------------------------------------*/

        ba::io_service *ioService;
        b::thread *signalThread;

        void handleSignals();
        static void signalHandler(ba::signal_set& signal, DoomController* controller, const bs::error_code& error, int sigNumber);
        void intSignal(int sigNumber);

        b::thread *doomThread;
        //bpr::child doomProcess;


        /* Message queues */
        /*------------------------------------------------------------------------------------------------------------*/

        struct Message {
            uint8_t code;
            char command[MQ_MAX_CMD_LEN];
        };

        void MQInit();
        void MQControllerSend(uint8_t code, const char *command = NULL);
        void MQDoomSend(uint8_t code, const char *command = NULL);
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
        bip::offset_t SMSize;
        std::string SMName;

        bip::mapped_region *InputSMRegion;
        InputState *input;
        InputState *_input;

        bip::mapped_region *GameStateSMRegion;
        GameState *gameState;

        bip::mapped_region *ScreenSMRegion;
        uint8_t *screen;


        /* Settings */
        /*------------------------------------------------------------------------------------------------------------*/

        unsigned int screenWidth, screenHeight, screenChannels, screenDepth;
        size_t screenPitch, screenSize;
        ScreenFormat screenFormat;
        bool depthBuffer;
        bool levelMap;
        //MapMode levelMapMode;
        bool labels;

        bool hud, weapon, crosshair, decals, particles;

        bool windowHidden, noXServer;

        bool noConsole;
        bool noSound;

        std::string exePath;
        std::string iwadPath;
        std::string filePath;
        std::string map;
        std::string demoPath;
        std::string configPath;
        int skill;

        bool allowDoomInput;
        bool runDoomAsync;

        unsigned int ticrate;
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
