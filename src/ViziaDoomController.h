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

#define DOOM_AMMO_Clip 0
#define DOOM_AMMO_Shell 1
#define DOOM_AMMO_Rocket 2
#define DOOM_AMMO_Cell 3

#define DOOM_WEAPON_Fist 0
#define DOOM_WEAPON_Chainsaw 0
#define DOOM_WEAPON_Pistol 1
#define DOOM_WEAPON_Shotgun 3
#define DOOM_WEAPON_SSG 3
#define DOOM_WEAPON_SuperShotgun 3
#define DOOM_WEAPON_Chaingun 4
#define DOOM_WEAPON_RocketLuncher 5
#define DOOM_WEAPON_Plasma 6
#define DOOM_WEAPON_BFG 7

#define V_ATTACK 0
#define V_USE 1
#define V_JUMP 2
#define V_CROUCH 3
//#define V_TURN180 4
#define V_ALTATTACK 5
#define V_RELOAD 6
#define V_ZOOM 7

#define V_SPEED 8
#define V_STRAFE 9

#define V_MOVERIGHT 10
#define V_MOVELEFT 11
#define V_BACK 12
#define V_FORWARD 13
#define V_RIGHT 14
#define V_LEFT 15
#define V_LOOKUP 16
#define V_LOOKDOWN 17
#define V_MOVEUP 18
#define V_MOVEDOWN 19
//#define V_SHOWSCORES 20

#define V_WEAPON1 21
#define V_WEAPON2 22
#define V_WEAPON3 23
#define V_WEAPON4 24
#define V_WEAPON5 25
#define V_WEAPON6 26
#define V_WEAPON7 27

#define V_WEAPONNEXT 28
#define V_WEAPONPREV 29

#define V_BT_SIZE 30

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

float DoomTic2S (unsigned int);

int DoomTic2Ms (unsigned int);

unsigned int S2DoomTic (float);

unsigned int Ms2DoomTic (float);

class ViziaDoomController {
    
    public:

        struct InputStruct{
            int MS_X;
            int MS_Y;
            bool BT[V_BT_SIZE];
        };

        struct GameVarsStruct{
            int GAME_TIC;

            int SCREEN_WIDTH;
            int SCREEN_HEIGHT;

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

            int PLAYER_AMMO[4];
            bool PLAYER_WEAPON[7];
            bool PLAYER_KEY[3];
        };

        ViziaDoomController();
        ~ViziaDoomController();

        //FLOW CONTROL

        bool init();
        bool close();
        bool tic(); bool update();
        void restartMap(); void resetMap();
        void restartGame();
        bool isDoomRunning();
        bool isRestartMapInLastTic();
        void sendCommand(std::string command);

        //SETTINGS

        //GAME & MAP SETTINGS

        void setInstanceId(std::string id);
        void setGamePath(std::string path);
        void setIwadPath(std::string path);
        void setFilePath(std::string path);
        void setMap(std::string map);
        void setSkill(int skill);

        void setAutoMapRestartOnTimeout(bool set);
        void setAutoMapRestartOnPlayerDeath(bool set);
        void setMapTimeout(unsigned int tics);

        //GRAPHIC SETTINGS

        void setScreenSize(int screenWidth, int screenHeight);
        void showHud(bool hud);
        void showWeapon(bool weapon);
        void showCrosshair(bool crosshair);
        void showDecals(bool decals);
        void showParticles(bool particles);

        void addViziaDoomStartArgument(std::string arg);

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

        int getGameTic();
        int getMapTic();

        int getMapKillCount();
        int getMapItemCount();
        int getMapSecretCount();

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

        struct MessageSignalStruct{
            uint8_t code;
        };

        struct MessageCommandStruct{
            uint8_t code;
            char command[VIZIA_MQ_MAX_CMD_LEN];
        };

        void SMInit();

        void SMSetSize(int screenWidth, int screenHeight);

        size_t SMGetInputRegionBeginning();
        size_t SMGetGameVarsRegionBeginning();
        size_t SMGetScreenRegionBeginning();

        void SMClose();

        void MQInit();

        void MQSend(uint8_t code);

        bool MQTrySend(uint8_t code);

        void MQSend(uint8_t code, const char * command);

        bool MQTrySend(uint8_t code, const char * command);

        void MQRecv(void *msg, unsigned long &size, unsigned int &priority);

        bool MQTryRecv(void *msg, unsigned long &size, unsigned int &priority);

        void MQClose();

        void waitForDoom();

        void lunchDoom();

        // OPTIONS

        unsigned int screenWidth, screenHeight, screenSize, colorDepth;

        bool hud, weapon, crosshair, decals, particles;

        std::string gamePath;
        std::string iwadPath;
        std::string file;
        std::string map;
        int skill;

        // AUTO RESTART

        bool autoRestartOnTimeout;
        bool autoRestartOnPlayersDeath;
        unsigned int mapTimeout;
        unsigned int mapRestartCount;
        bool mapRestarting;

        // COMMUNICATION

        void generateInstanceId();

        std::string instanceId;

        bip::shared_memory_object SM;
        size_t SMSize;
        std::string SMName;

        bip::message_queue *MQController;
        bip::message_queue *MQDoom;
        std::string MQControllerName;
        std::string MQDoomName;

        bip::mapped_region *InputSMRegion;
        InputStruct *Input;

        bip::mapped_region *GameVarsSMRegion;
        GameVarsStruct *GameVars;

        bip::mapped_region *ScreenSMRegion;
        uint8_t *Screen;

        b::thread *doomThread;
        //bpr::child doomProcess;
        bool doomRunning;
        bool doomTic;
};


#endif
