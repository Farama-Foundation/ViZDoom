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

#include "ViZDoomTypes.h"
#include "ViZDoomMessageQueue.h"
#include "ViZDoomSharedMemory.h"

#include <boost/asio.hpp>
#include <boost/random.hpp>
#include <boost/thread.hpp>
#include <string>
#include <vector>

namespace vizdoom {

    namespace b         = boost;
    namespace ba        = boost::asio;
    namespace bip       = boost::interprocess;
    namespace br        = boost::random;
    namespace bs        = boost::system;

#define INSTANCE_ID_LENGHT 10

/* Shared memory's settings */
#define SM_NAME_BASE        "ViZDoomSM"

/* Message queues' settings */
#define MQ_CTR_NAME_BASE    "ViZDoomMQCtr"
#define MQ_DOOM_NAME_BASE   "ViZDoomMQDoom"

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
        void restartMap(std::string demoPath = "");
        void respawnPlayer();
        bool isDoomRunning();
        void sendCommand(std::string command);

        void setTicrate(unsigned int ticrate);
        unsigned int getTicrate();

        unsigned int getDoomSeed();
        void setDoomSeed(unsigned int seed);
        void clearDoomSeed();

        unsigned int getInstanceSeed();
        void setInstanceSeed(unsigned int seed);

        std::string getMap();
        void setMap(std::string map, std::string demoPath = "");
        void playDemo(std::string demoPath, int player = 0);


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
        void setRenderMinimalHud(bool minHud);
        void setRenderWeapon(bool weapon);
        void setRenderCrosshair(bool crosshair);
        void setRenderDecals(bool decals);
        void setRenderParticles(bool particles);
        void setRenderEffectsSprites(bool sprites);
        void setRenderMessages(bool messages);
        void setRenderCorpses(bool corpses);
        void setRenderScreenFlashes(bool flashes);

        void setScreenResolution(unsigned int width, unsigned int height);
        unsigned int getScreenWidth();
        void setScreenWidth(unsigned int width);
        unsigned int getScreenHeight();
        void setScreenHeight(unsigned int height);
        ScreenFormat getScreenFormat();

        void setScreenFormat(ScreenFormat format);
        unsigned int getScreenChannels();
        unsigned int getScreenDepth();
        size_t getScreenPitch();
        size_t getScreenSize();

        /* Depth buffer */
        bool isDepthBufferEnabled();
        void setDepthBufferEnabled(bool depthBuffer);

        /* Labels */
        bool isLabelsEnabled();
        void setLabelsEnabled(bool labels);

        /* Automap */
        bool isAutomapEnabled();
        void setAutomapEnabled(bool map);
        void setAutomapMode(AutomapMode mode);
        void setAutomapRotate(bool rotate);
        void setAutomapRenderTextures(bool textures);

        /* Buffers in SM */
        uint8_t *const getScreenBuffer();
        uint8_t *const getDepthBuffer();
        uint8_t *const getLabelsBuffer();
        uint8_t *const getAutomapBuffer();

        /* Buttons getters and setters */
        /*------------------------------------------------------------------------------------------------------------*/

        SMInputState *const getInput();

        SMGameState *const getGameState();

        /* Buttons state */
        int getButtonState(Button button);
        void setButtonState(Button button, int state);
        void toggleButtonState(Button button);

        /* Buttons availableity */
        bool isButtonAvailable(Button button);
        void setButtonAvailable(Button button, bool set);
        void resetButtons();
        void disableAllButtons();

        int getButtonMaxValue(Button button);
        void setButtonMaxValue(Button button, unsigned int value);
        void availableAllButtons();
        bool isAllowDoomInput();
        void setAllowDoomInput(bool set);
        bool isRunDoomAsync();
        void setRunDoomAsync(bool set);


        /* GameState getters */
        /*------------------------------------------------------------------------------------------------------------*/

        double getGameVariable(GameVariable var);
        unsigned int getGameTic();
        bool isMultiplayerGame();
        bool isNetGame();
        unsigned int getMapTic();
        int getMapReward();
        bool isPlayerDead();


    private:

        /* Flow */
        /*------------------------------------------------------------------------------------------------------------*/

        bool doomRunning;
        bool doomWorking;

        bool receiveMQMsg();
        void waitForDoomStart();
        void waitForDoomWork();
        void waitForDoomMapStartTime();
        void createDoomArgs();
        void launchDoom();

        /* Seed */
        /*------------------------------------------------------------------------------------------------------------*/

        void generateInstanceId();

        unsigned int getNextDoomSeed();

        void forceDoomSeed(unsigned int seed);

        bool doomStaticSeed;
        unsigned int doomSeed;
        unsigned int instanceSeed;

        br::mt19937 instanceRng;
        std::string instanceId;


        /* Threads */
        /*------------------------------------------------------------------------------------------------------------*/

        ba::io_service *ioService;
        b::thread *signalThread;

        void handleSignals();

        static void signalHandler(ba::signal_set &signal, DoomController *controller,
                                  const bs::error_code &error, int sigNumber);

        void intSignal(int sigNumber);

        b::thread *doomThread;
        //bpr::child doomProcess;


        /* Message queues */
        /*------------------------------------------------------------------------------------------------------------*/

        MessageQueue *MQDoom;
        MessageQueue *MQController;


        /* Shared memory */
        /*------------------------------------------------------------------------------------------------------------*/

        SharedMemory *SM;

        SMGameState *gameState;

        SMInputState *input;
        SMInputState *_input;

        uint8_t *screenBuffer;
        uint8_t *depthBuffer;
        uint8_t *automapBuffer;
        uint8_t *labelsBuffer;


        /* Settings */
        /*------------------------------------------------------------------------------------------------------------*/

        unsigned int screenWidth, screenHeight, screenChannels, screenDepth;
        size_t screenPitch, screenSize;
        ScreenFormat screenFormat;
        bool depth;
        bool automap;
        bool labels;

        bool hud, minHud, weapon, crosshair, decals, particles, sprites, messages, corpses, flashes;
        AutomapMode amMode;
        bool amRotate, amTextures;

        bool updateSettings;

        int getRenderModeValue();

        void setRenderMode(int value);

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
        bool mapChanging;
        unsigned int mapLastTic;

        std::vector<std::string> customArgs;
        std::vector<std::string> doomArgs;

    };

}

#endif
