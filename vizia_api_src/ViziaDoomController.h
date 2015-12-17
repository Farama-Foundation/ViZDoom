#ifndef __VIZIA_DOOM_CONTROLLER_H__
#define __VIZIA_DOOM_CONTROLLER_H__

#include "ViziaDefines.h"

#include <string>

#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include "boost/process.hpp"

namespace Vizia{

    namespace b = boost;
    namespace bip = boost::interprocess;
    namespace bpr = boost::process;
    namespace bpri = boost::process::initializers;

#define SM_NAME_BASE "ViziaSM"

#define MQ_NAME_CTR_BASE "ViziaMQCtr"
#define MQ_NAME_DOOM_BASE "ViziaMQDoom"
#define MQ_MAX_MSG_NUM 32
#define MQ_MAX_MSG_SIZE sizeof(DoomController::MessageCommandStruct)
#define MQ_MAX_CMD_LEN 32

#define MSG_CODE_DOOM_DONE 11
#define MSG_CODE_DOOM_CLOSE 12
#define MSG_CODE_DOOM_ERROR 13

#define MSG_CODE_TIC 21
#define MSG_CODE_UPDATE 22
#define MSG_CODE_TIC_N_UPDATE 23
#define MSG_CODE_COMMAND 24
#define MSG_CODE_CLOSE 25
#define MSG_CODE_ERROR 26

    class DoomController {

    public:

        struct InputStruct {
            int MS_X;
            int MS_Y;
            int MS_MAX_X;
            int MS_MAX_Y;
            bool BT[ButtonsNumber];
            bool BT_AVAILABLE[ButtonsNumber];
        };

        struct GameVarsStruct {
            unsigned int GAME_TIC;
            unsigned int GAME_SEED;
            unsigned int GAME_STAIC_SEED;

            unsigned int SCREEN_WIDTH;
            unsigned int SCREEN_HEIGHT;
            size_t SCREEN_PITCH;
            size_t SCREEN_SIZE;
            int SCREEN_FORMAT;

            unsigned int MAP_START_TIC;
            unsigned int MAP_TIC;

            int MAP_REWARD;

            int MAP_USER_VARS[UserVarsNumber];

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

            bool PLAYER_WEAPON_READY;

            int PLAYER_SELECTED_WEAPON;
            int PLAYER_SELECTED_WEAPON_AMMO;

            int PLAYER_AMMO[4];
            bool PLAYER_WEAPON[7];
            bool PLAYER_KEY[3];
        };

        DoomController();
        ~DoomController();

        //FLOW CONTROL

        bool init();
        void close();
        void restart();

        bool tic();
        bool tic(bool update);
        bool tics(unsigned int tics);
        bool tics(unsigned int tics, bool update);
        void restartMap();
        void resetMap();
        bool isDoomRunning();
        void sendCommand(std::string command);

        //SETTINGS

        //GAME & MAP SETTINGS

        unsigned int getCurrentSeed();
        unsigned int getSeed();
        void setSeed(unsigned int seed);

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

        int getScreenWidth();
        int getScreenHeight();
        int getScreenChannels();
        size_t getScreenPitch();
        size_t getScreenSize();

        ScreenFormat getScreenFormat();

        //PUBLIC SETTERS & GETTERS

        uint8_t *const getScreen();

        InputStruct *const getInput();

        GameVarsStruct *const getGameVars();

        void setMouse(int x, int y);
        int getMouseX();
        void setMouseX(int x);
        int getMouseY();
        void setMouseY(int y);
        void resetMouse();

        bool getButtonState(Button button);
        void setButtonState(Button button, bool state);
        void toggleButtonState(Button button);
        bool isButtonAvailable(Button button);
        void setButtonAvailable(Button button, bool set);
        void resetButtons();
        void resetDescreteButtons();
        void disableAllButtons();
        void availableAllButtons();

        void resetInput();

        bool isAllowDoomInput();
        void setAllowDoomInput(bool set);

        int getGameVar(GameVar var);

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
        int getPlayerSelectedWeaponAmmo();
        int getPlayerSelectedWeapon();

        int getPlayerAmmo1();
        int getPlayerAmmo2();
        int getPlayerAmmo3();
        int getPlayerAmmo4();

        bool getPlayerWeapon1();
        bool getPlayerWeapon2();
        bool getPlayerWeapon3();
        bool getPlayerWeapon4();
        bool getPlayerWeapon5();
        bool getPlayerWeapon6();
        bool getPlayerWeapon7();

        bool getPlayerKey1();
        bool getPlayerKey2();
        bool getPlayerKey3();

    private:

        void generateSeed();
        void generateInstanceId();

        int seed;
        std::string instanceId;

        b::thread *doomThread;
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
        void MQSend(uint8_t code);
        void MQSelfSend(uint8_t code);
        bool MQTrySend(uint8_t code);
        void MQSend(uint8_t code, const char *command);
        bool MQTrySend(uint8_t code, const char *command);
        void MQRecv(void *msg, unsigned long &size, unsigned int &priority);
        bool MQTryRecv(void *msg, unsigned long &size, unsigned int &priority);
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
        InputStruct *Input;

        bip::mapped_region *GameVarsSMRegion;
        GameVarsStruct *GameVars;

        bip::mapped_region *ScreenSMRegion;
        uint8_t *Screen;

        //HELPERS

        void waitForDoomStart();
        void waitForDoomWork();
        void waitForDoomMapStartTime();
        void lunchDoom();

        // OPTIONS

        unsigned int screenWidth, screenHeight, screenChannels;
        size_t screenPitch, screenSize;
        ScreenFormat screenFormat;

        bool hud, weapon, crosshair, decals, particles;

        bool windowHidden, noXServer;

        std::string gamePath;
        std::string iwadPath;
        std::string filePath;
        std::string map;
        std::string configPath;
        int skill;

        bool allowDoomInput;

        // AUTO RESTART

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

    };

}

#endif
