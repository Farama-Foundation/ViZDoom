#ifndef __VIZIA_DOOM_CONTROLLER_H__
#define __VIZIA_DOOM_CONTROLLER_H__

#include "ViziaDefines.h"

#include <ctime>
#include <string>

#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/chrono/chrono.hpp>
#include <boost/asio.hpp>
#include "boost/process.hpp"

namespace Vizia{

    namespace b = boost;
    namespace bip = boost::interprocess;
    namespace bpr = boost::process;
    namespace bpri = boost::process::initializers;
    namespace bc = boost::chrono;
    namespace bs = boost::system;
    namespace ba = boost::asio;

#define SM_NAME_BASE "ViziaSM"

#define MQ_NAME_CTR_BASE "ViziaMQCtr"
#define MQ_NAME_DOOM_BASE "ViziaMQDoom"
#define MQ_MAX_MSG_NUM 64
#define MQ_MAX_MSG_SIZE sizeof(DoomController::MessageCommandStruct)
#define MQ_MAX_CMD_LEN 64

#define MSG_CODE_DOOM_DONE 11
#define MSG_CODE_DOOM_CLOSE 12
#define MSG_CODE_DOOM_ERROR 13
#define MSG_CODE_DOOM_PROCESS_EXIT 14

#define MSG_CODE_TIC 21
#define MSG_CODE_UPDATE 22
#define MSG_CODE_TIC_N_UPDATE 23
#define MSG_CODE_COMMAND 24
#define MSG_CODE_CLOSE 25
#define MSG_CODE_ERROR 26

#define MSG_CODE_SIGNAL_INT_ABRT_TERM 30

    class DoomController {

    public:

        struct InputStruct {
            int BT[ButtonsNumber];
            bool BT_AVAILABLE[ButtonsNumber];
            int BT_MAX_VALUE[AxisButtonsNumber];
        };

        struct GameVariablesStruct {
            unsigned int GAME_TIC;
            unsigned int GAME_SEED;
            unsigned int GAME_STATIC_SEED;

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

            bool PLAYER_DEAD;

            int PLAYER_KILLCOUNT;
            int PLAYER_ITEMCOUNT;
            int PLAYER_SECRETCOUNT;
            int PLAYER_FRAGCOUNT; //in multi

            bool PLAYER_ON_GROUND;

            int PLAYER_HEALTH;
            int PLAYER_ARMOR;

            bool PLAYER_ATTACK_READY;
            bool PLAYER_ALTATTACK_READY;

            int PLAYER_SELECTED_WEAPON;
            int PLAYER_SELECTED_WEAPON_AMMO;

            int PLAYER_AMMO[SlotsNumber];
            int PLAYER_WEAPON[SlotsNumber];
        };

        DoomController();
        ~DoomController();

        //FLOW CONTROL

        bool init();
        void close();
        void restart();

        void intSignal();

        bool tic();
        bool tic(bool update);
        bool tics(unsigned int tics);
        bool tics(unsigned int tics, bool update);
        void restartMap();
        void resetMap();
        bool isDoomRunning();
        void sendCommand(std::string command);
        void addCustomArg(std::string arg);
        void clearCustomArgs();

        void waitRealTimeForTics(unsigned int tics);

        //SETTINGS

        //GAME & MAP SETTINGS

        unsigned int getSeed();
        unsigned int getStaticSeed();
        void setStaticSeed(unsigned int seed);
        void setUseStaticSeed(bool use);
        bool isUseStaticSeed();

        std::string getInstanceId();
        void setInstanceId(std::string id);

        std::string getGamePath();
        void setGamePath(std::string path);

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

        void setAutoMapRestart(bool set);
        void setAutoMapRestartOnTimeout(bool set);
        void setAutoMapRestartOnPlayerDeath(bool set);
        void setAutoMapRestartOnMapEnd(bool set);

        unsigned int getMapStartTime();
        void setMapStartTime(unsigned int tics);

        unsigned int getMapTimeout();
        void setMapTimeout(unsigned int tics);

        bool isMapLastTic();
        bool isMapFirstTic();
        bool isMapEnded();

        void setNoConsole(bool console);

        //GRAPHIC SETTINGS

        void setWindowHidden(bool windowHidden);
        void setNoXServer(bool noXServer);

        void setScreenResolution(unsigned int width, unsigned int height);
        void setScreenWidth(unsigned int width);
        void setScreenHeight(unsigned int height);
        void setScreenFormat(ScreenFormat format);

        void setRenderHud(bool hud);
        void setRenderWeapon(bool weapon);
        void setRenderCrosshair(bool crosshair);
        void setRenderDecals(bool decals);
        void setRenderParticles(bool particles);

        unsigned int getScreenWidth();
        unsigned int getScreenHeight();
        unsigned int getScreenChannels();
        unsigned int getScreenDepth();
        size_t getScreenPitch();
        size_t getScreenSize();

        ScreenFormat getScreenFormat();

        //PUBLIC SETTERS & GETTERS

        uint8_t * const getScreen();
        InputStruct * const getInput();
        GameVariablesStruct * const getGameVariables();

        int getButtonState(Button button);
        void setButtonState(Button button, int state);
        void toggleButtonState(Button button);
        bool isButtonAvailable(Button button);
        void setButtonAvailable(Button button, bool set);
        void resetButtons();
        void resetDescreteButtons();
        void disableAllButtons();
        void setButtonMaxValue(Button button, int value);
        int getButtonMaxValue(Button button);
        void availableAllButtons();
        bool isButtonDiscrete(Button button);
        bool isButtonAxis(Button button);

        bool isAllowDoomInput();
        void setAllowDoomInput(bool set);

        bool isRunDoomAsync();
        void setRunDoomAsync(bool set);

        int getGameVariable(GameVariable var);

        int getGameTic();
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

        int getPlayerHealth();
        int getPlayerArmor();
        bool isPlayerOnGround();
        bool isPlayerAttackReady();
        bool isPlayerAltAttackReady();
        int getPlayerSelectedWeaponAmmo();
        int getPlayerSelectedWeapon();

        int getPlayerAmmo(unsigned int slot);
        int getPlayerWeapon(unsigned int slot);

    private:

        void generateStaticSeed();
        void generateInstanceId();

        bool useStaticSeed;
        int staticSeed;
        std::string instanceId;

        b::thread *doomThread;
        ba::io_service ioService;
        b::thread *signalThread;
        //bpr::child doomProcess;
        bool doomRunning;
        bool doomWorking;

        //MESSAGE QUEUES

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
        void MQControllerRecv(void *msg, unsigned long &size, unsigned int &priority);
        void MQClose();

        bip::message_queue *MQController;
        bip::message_queue *MQDoom;
        std::string MQControllerName;
        std::string MQDoomName;

        //SHARED MEMORY

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

        //HELPERS

        void waitForDoomStart();
        void waitForDoomWork();
        void waitForDoomMapStartTime();
        void launchDoom();
        void handleSignals();

        // OPTIONS

        unsigned int screenWidth, screenHeight, screenChannels, screenDepth;
        size_t screenPitch, screenSize;
        ScreenFormat screenFormat;

        bool hud, weapon, crosshair, decals, particles;

        bool windowHidden, noXServer;

        bool noConsole;

        std::string gamePath;
        std::string iwadPath;
        std::string filePath;
        std::string map;
        std::string configPath;
        int skill;

        bool allowDoomInput;
        bool runDoomAsync;

        // AUTO RESTART & MAP TIME

        bool autoRestart;
        bool autoRestartOnTimeout;
        bool autoRestartOnPlayersDeath;
        bool autoRestartOnMapEnd;

        unsigned int mapStartTime;
        unsigned int mapTimeout;
        unsigned int mapRestartCount;
        bool mapRestarting;
        bool mapEnded;
        unsigned int mapLastTic;

        // TIME

        //bc::steady_clock::time_point lastTicTime;
        std::clock_t lastTicTime;

        //CUSTOM ARGS

        std::vector <std::string> customArgs;

    };

}

#endif
