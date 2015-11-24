#ifndef __VIZIA_DOOM_CONTROLLER_H__
#define __VIZIA_DOOM_CONTROLLER_H__

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

#define DOOM_AMMO_CLIP 0
#define DOOM_AMMO_SHELL 1
#define DOOM_AMMO_ROCKET 2
#define DOOM_AMMO_CELL 3

#define DOOM_WEAPON_FIST 0
#define DOOM_WEAPON_CHAINSAW 0
#define DOOM_WEAPON_PISTOL 1
#define DOOM_WEAPON_SHOTGUN 3
#define DOOM_WEAPON_SSG 3
#define DOOM_WEAPON_SUPER_SHOTGUN 3
#define DOOM_WEAPON_CHAINGUN 4
#define DOOM_WEAPON_ROCKET_LUNCHER 5
#define DOOM_WEAPON_PLASMA_GUN 6
#define DOOM_WEAPON_BFG 7

#define DOOM_KEY_BLUE 0
#define DOOM_KEY_RED 1
#define DOOM_KEY_YELLOW 2

#define A_ATTACK 0
#define A_USE 1
#define A_JUMP 2
#define A_CROUCH 3
#define A_TURN180 4
#define A_ALTATTACK 5
#define A_RELOAD 6
#define A_ZOOM 7

#define A_SPEED 8
#define A_STRAFE 9

#define A_MOVERIGHT 10
#define A_MOVELEFT 11
#define A_BACK 12
#define A_FORWARD 13
#define A_RIGHT 14
#define A_LEFT 15
#define A_LOOKUP 16
#define A_LOOKDOWN 17
#define A_MOVEUP 18
#define A_MOVEDOWN 19
//#define A_SHOWSCORES 20

#define A_WEAPON1 21
#define A_WEAPON2 22
#define A_WEAPON3 23
#define A_WEAPON4 24
#define A_WEAPON5 25
#define A_WEAPON6 26
#define A_WEAPON7 27

#define A_WEAPONNEXT 28
#define A_WEAPONPREV 29

#define A_BT_SIZE 30

#define V_KILLCOUNT 0
#define V_ITEMCOUNT 1
#define V_SECRETCOUNT 2
#define V_HEALTH 3
#define V_ARMOR 4
#define V_SELECTED_WEAPON 5
#define V_SELECTED_WEAPON_AMMO 6
#define V_AMMO1 7
#define V_AMMO2 8
#define V_AMMO3 9
#define V_AMMO4 10
#define V_WEAPON1 11
#define V_WEAPON2 12
#define V_WEAPON3 13
#define V_WEAPON4 14
#define V_WEAPON5 15
#define V_WEAPON6 16
#define V_WEAPON7 17
#define V_KEY1 18
#define V_KEY2 19
#define V_KEY3 20
#define V_SIZE 21

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

#define VIZIA_SCREEN_CRCGCB 0
#define VIZIA_SCREEN_CRCGCBCA 1
#define VIZIA_SCREEN_RGB24 2
#define VIZIA_SCREEN_RGBA32 3
#define VIZIA_SCREEN_ARGB32 4
#define VIZIA_SCREEN_CBCGCR 5
#define VIZIA_SCREEN_CBCGCRCA 6
#define VIZIA_SCREEN_BGR24 7
#define VIZIA_SCREEN_BGRA32 8
#define VIZIA_SCREEN_ABGR32 9

class ViziaDoomController {
    
    public:

        struct InputStruct{
            int MS_X;
            int MS_Y;
            int MS_MAX_X;
            int MS_MAX_Y;
            bool BT[A_BT_SIZE];
            bool BT_AVAILABLE[A_BT_SIZE];
        };

        struct GameVarsStruct{
            int GAME_TIC;

            int SCREEN_WIDTH;
            int SCREEN_HEIGHT;
            int SCREEN_PITCH;
            int SCREEN_SIZE;
            int SCREEN_FORMAT;

            int MAP_REWARD;;

            int MAP_START_TIC;
            int MAP_TIC;

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

        static int getButtonId(std::string name);
        static int getGameVarId(std::string name);

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
        void setMapTimeout(unsigned int tics);
        bool isMapLastTic();
        bool isMapFirstTic();

        //GRAPHIC SETTINGS

        void setScreenResolution(int width, int height);
        void setScreenWidth(int width);
        void setScreenHeight(int height);
        void setScreenFormat(int format);
        void setRenderHud(bool hud);
        void setRenderWeapon(bool weapon);
        void setRenderCrosshair(bool crosshair);
        void setRenderDecals(bool decals);
        void setRenderParticles(bool particles);

        int getScreenWidth();
        int getScreenHeight();
        int getScreenPitch();
        int getScreenSize();
        int getScreenFormat();

        //PUBLIC SETTERS & GETTERS

        uint8_t* const getScreen();
        InputStruct* const getInput();
        GameVarsStruct* const getGameVars();

        void setMouse(int x, int y);
        void setMouseX(int x);
        void setMouseY(int y);
        void setButtonState(int button, bool state);
        void setKeyState(int key, bool state);
        void toggleButtonState(int button);
        void toggleKeyState(int key);
        void resetInput();

        int getGameVar(int var);

        int getGameTic();

        int getMapReward();
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

        void waitForDoom();

        void lunchDoom();

        // OPTIONS

        unsigned int screenWidth, screenHeight, screenPitch, screenSize, screenFormat;

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
        unsigned int mapTimeout;
        unsigned int mapRestartCount;
        bool mapRestarting;
        bool mapEnded;
        int mapLastTic;

};


#endif
