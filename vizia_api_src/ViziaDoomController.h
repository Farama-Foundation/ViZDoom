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

namespace b = boost;
namespace bip = boost::interprocess;
namespace bpr = boost::process;
namespace bpri = boost::process::initializers;

#define VIZIA_SM_NAME_BASE "ViziaSM"

#define VIZIA_MQ_NAME_CTR_BASE "ViziaMQCtr"
#define VIZIA_MQ_NAME_DOOM_BASE "ViziaMQDoom"
#define VIZIA_MQ_MAX_MSG_NUM 32
#define VIZIA_MQ_MAX_MSG_SIZE sizeof(ViziaDoomController::MessageCommandStruct)
#define VIZIA_MQ_MAX_CMD_LEN 32

#define VIZIA_MSG_CODE_DOOM_READY 10
#define VIZIA_MSG_CODE_DOOM_TIC 11
#define VIZIA_MSG_CODE_DOOM_CLOSE 12

#define VIZIA_MSG_CODE_READY 0
#define VIZIA_MSG_CODE_TIC 1
#define VIZIA_MSG_CODE_CLOSE 2
#define VIZIA_MSG_CODE_COMMAND 3

class ViziaDoomController {
    
    public:

        struct InputStruct{
            int MS_X;
            int MS_Y;
            int MS_MAX_X;
            int MS_MAX_Y;
            bool BT[ViziaButtonsNumber];
            bool BT_AVAILABLE[ViziaButtonsNumber];
        };

        struct GameVarsStruct{
            unsigned int GAME_TIC;

            unsigned int SCREEN_WIDTH;
            unsigned int SCREEN_HEIGHT;
            size_t SCREEN_PITCH;
            size_t SCREEN_SIZE;
            int SCREEN_FORMAT;

            int MAP_REWARD;
            int SHAPING_REWARD;

            unsigned int MAP_START_TIC;
            unsigned int MAP_TIC;

            int MAP_KILLCOUNT;
            int MAP_ITEMCOUNT;
            int MAP_SECRETCOUNT;
            bool MAP_END;

            bool PLAYER_DEAD;

            int PLAYER_KILLCOUNT;
            int PLAYER_ITEMCOUNT;
            int PLAYER_SECRETCOUNT;
            int PLAYER_FRAGCOUNT; //for multiplayer

            bool PLAYER_ONGROUND;

            int PLAYER_HEALTH;
            int PLAYER_ARMOR;

            int PLAYER_SELECTED_WEAPON;
            int PLAYER_SELECTED_WEAPON_AMMO;

            int PLAYER_AMMO[10];
            bool PLAYER_WEAPON[10];
            bool PLAYER_KEY[10];
        };

        static ViziaButton getButtonId(std::string name);
        static ViziaGameVar getGameVarId(std::string name);

        ViziaDoomController();
        ~ViziaDoomController();

        //FLOW CONTROL

        bool init();
        bool close();
        bool tic(); bool update();
        void restartMap(); void resetMap();
        void restartGame();
        bool isDoomRunning();
        void sendCommand(std::string command);
        void resetConfig();

        //SETTINGS

        //GAME & MAP SETTINGS

        void setInstanceId(std::string id);
        void setGamePath(std::string path);
        void setIwadPath(std::string path);
        void setFilePath(std::string path);
        void setMap(std::string map);
        void setSkill(int skill);
        void setConfigPath(std::string path);

        void setAutoMapRestart(bool set);
        void setAutoMapRestartOnTimeout(bool set);
        void setAutoMapRestartOnPlayerDeath(bool set);
        void setAutoMapRestartOnMapEnd(bool set);
        void setMapTimeout(unsigned int tics);
        bool isMapLastTic();
        bool isMapFirstTic();
        bool isMapEnded();

        //GRAPHIC SETTINGS

        void setScreenResolution(int width, int height);
        void setScreenWidth(int width);
        void setScreenHeight(int height);
        void setScreenFormat(ViziaScreenFormat format);
        void setRenderHud(bool hud);
        void setRenderWeapon(bool weapon);
        void setRenderCrosshair(bool crosshair);
        void setRenderDecals(bool decals);
        void setRenderParticles(bool particles);

        int getScreenWidth();
        int getScreenHeight();
        size_t getScreenPitch();
        size_t getScreenSize();
        ViziaScreenFormat getScreenFormat();

        //PUBLIC SETTERS & GETTERS

        uint8_t* const getScreen();
        InputStruct* const getInput();
        GameVarsStruct* const getGameVars();

        void setMouse(int x, int y);
        void setMouseX(int x);
        void setMouseY(int y);
        void setButtonState(ViziaButton button, bool state);
        void toggleButtonState(ViziaButton button);
        void setAllowButton(ViziaButton button, bool allow);
        void allowAllButtons();
        void resetInput();

        int getGameVar(ViziaGameVar var);

        int getGameTic();

        int getMapReward();
        int getShapingReward();

        int getMapTimeout();

        int getMapTic();

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

        void generateInstanceId();
        std::string instanceId;

        b::thread *doomThread;
        //bpr::child doomProcess;
        bool doomRunning;
        bool doomTic;

        //MESSAGE QUEUES

        struct MessageSignalStruct{
            uint8_t code;
        };

        struct MessageCommandStruct{
            uint8_t code;
            char command[VIZIA_MQ_MAX_CMD_LEN];
        };

        void MQInit();

        void MQSend(uint8_t code);

        bool MQTrySend(uint8_t code);

        void MQSend(uint8_t code, const char * command);

        bool MQTrySend(uint8_t code, const char * command);

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
        void waitForDoomTic();

        void lunchDoom();

        // OPTIONS

        unsigned int screenWidth, screenHeight;
        size_t screenPitch, screenSize;
        ViziaScreenFormat screenFormat;

        bool hud, weapon, crosshair, decals, particles;

        std::string gamePath;
        std::string iwadPath;
        std::string filePath;
        std::string map;
        std::string configPath;
        int skill;

        // AUTO RESTART

        bool autoRestart;
        bool autoRestartOnTimeout;
        bool autoRestartOnPlayersDeath;
        bool autoRestartOnMapEnd;
        unsigned int mapTimeout;
        unsigned int mapRestartCount;
        bool mapRestarting;
        bool mapEnded;
        unsigned int mapLastTic;

};


#endif
