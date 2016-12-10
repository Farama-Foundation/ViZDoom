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

#include "ViZDoomController.h"
#include "ViZDoomExceptions.h"
#include "ViZDoomPathHelpers.h"
#include "ViZDoomUtilities.h"
#include "ViZDoomVersion.h"
#include "boost/process.hpp"

#include <boost/algorithm/string.hpp>
#include <boost/chrono.hpp>
#include <boost/lexical_cast.hpp>


namespace vizdoom {

    namespace bal       = boost::algorithm;
    namespace bc        = boost::chrono;
    namespace bfs       = boost::filesystem;
    namespace bpr       = boost::process;
    namespace bpri      = boost::process::initializers;


    /* Public methods */
    /*----------------------------------------------------------------------------------------------------------------*/

    DoomController::DoomController() {
        /* Message queues */
        this->MQController = nullptr;
        this->MQDoom = nullptr;

        /* Shared memory */
        this->SM = nullptr;
        this->gameState = nullptr;
        this->input = nullptr;
        this->screenBuffer = nullptr;
        this->depthBuffer = nullptr;
        this->labelsBuffer = nullptr;
        this->automapBuffer = nullptr;

        /* Threads */
        this->signalThread = nullptr;
        this->doomThread = nullptr;

        /* Flow control */
        this->doomRunning = false;
        this->doomWorking = false;

        this->mapStartTime = 1;
        this->mapTimeout = 0;
        this->mapRestartCount = 0;
        this->mapChanging = false;
        this->mapLastTic = 1;

        // TODO move default settings to separate file (ViZDoomConsts.h perhaps?
        /* Settings */
        this->ticrate = DEFAULT_TICRATE;

        this->exePath = "";
        this->iwadPath = "";
        this->filePath = "";
        this->map = "map01";
        this->demoPath = "";
        this->configPath = "";
        this->skill = 3;

        this->screenWidth = 320;
        this->screenHeight = 240;
        this->screenChannels = 3;
        this->screenPitch = 320;
        this->screenSize = this->screenWidth * this->screenHeight;
        this->screenDepth = 8;
        this->screenFormat = CRCGCB;

        this->depth = false;

        this->labels = false;

        this->automap = false;
        this->amMode = NORMAL;
        this->amRotate = false;
        this->amTextures = true;

        this->hud = false;
        this->minHud = false;
        this->weapon = true;
        this->crosshair = false;
        this->decals = true;
        this->particles = true;
        this->sprites = true;
        this->messages = false;
        this->corpses = true;

        this->windowHidden = false;
        this->noXServer = false;
        this->noConsole = true;
        this->noSound = true;

        this->allowDoomInput = false;
        this->runDoomAsync = false;

        this->doomStaticSeed = true;
        this->doomSeed = 0;

        this->instanceRng.seed(static_cast<unsigned int>(bc::high_resolution_clock::now().time_since_epoch().count()));

        this->_input = new SMInputState();
    }

    DoomController::~DoomController() {
        this->close();
        delete _input;
    }


    /* Flow Control */
    /*----------------------------------------------------------------------------------------------------------------*/

    bool DoomController::init() {

        if (!this->doomRunning) {

            try {
                this->generateInstanceId();

                // Generate Doom process's arguments
                this->createDoomArgs();

                // Create message queues
                this->MQDoom = new MessageQueue(MQ_DOOM_NAME_BASE + this->instanceId);
                this->MQController = new MessageQueue(MQ_CTR_NAME_BASE + this->instanceId);

                // Signal handle thread
                this->signalThread = new b::thread(b::bind(&DoomController::handleSignals, this));

                // Doom thread
                this->doomThread = new b::thread(b::bind(&DoomController::launchDoom, this));
                this->doomRunning = true;

                // Wait for first message from Doom
                this->waitForDoomStart();

                // Open shared memory
                this->SM = new SharedMemory(SM_NAME_BASE + this->instanceId);

                this->gameState = this->SM->getGameState();
                this->input = this->SM->getInputState();
                this->screenBuffer = this->SM->getScreenBuffer();
                this->depthBuffer = this->SM->getDepthBuffer();
                this->labelsBuffer = this->SM->getLabelsBuffer();
                this->automapBuffer = this->SM->getAutomapBuffer();

                // Check version
                if (this->gameState->VERSION != VIZDOOM_LIB_VERSION)
                    throw ViZDoomErrorException(
                            std::string("Controlled ViZDoom version (") + this->gameState->VERSION_STR +
                            ") does not match library version (" + VIZDOOM_LIB_VERSION_STR + ").");

                this->waitForDoomMapStartTime();

                // Update state
                this->MQDoom->send(MSG_CODE_UPDATE);
                this->waitForDoomWork();

                *this->input = *this->_input;

                this->mapLastTic = this->gameState->MAP_TIC;

            }
            catch (...) {
                this->close();
                throw;
            }
        }

        return this->doomRunning;
    }

    void DoomController::close() {

        if (this->doomRunning) {

            this->doomRunning = false;
            this->doomWorking = false;

            this->MQDoom->send(MSG_CODE_CLOSE);
        }

        if (this->signalThread && this->signalThread->joinable()) {
            this->ioService->stop();

            this->signalThread->interrupt();
            this->signalThread->join();
            delete this->signalThread;
            this->signalThread = nullptr;

            delete this->ioService;
            this->ioService = nullptr;
        }

        if (this->doomThread && this->doomThread->joinable()) {
            this->doomThread->interrupt();
            this->doomThread->join();
            delete this->doomThread;
            this->doomThread = nullptr;
        }

        if (this->SM) {
            delete this->SM;
            this->SM = nullptr;
        }

        if (this->MQDoom) {
            delete this->MQDoom;
            this->MQDoom = nullptr;
        }
        if (this->MQController) {
            delete this->MQController;
            this->MQController = nullptr;
        }

        this->gameState = nullptr;
        this->input = nullptr;
        this->screenBuffer = nullptr;
        this->depthBuffer = nullptr;
        this->labelsBuffer = nullptr;
        this->automapBuffer = nullptr;
    }

    void DoomController::restart() {
        this->close();
        this->init();
    }

    bool DoomController::isTicPossible() {
        return !((!this->gameState->GAME_MULTIPLAYER && this->gameState->PLAYER_DEAD)
                 || (this->mapTimeout > 0 && this->gameState->MAP_TIC >= this->mapTimeout + this->mapStartTime)
                 || (this->gameState->MAP_END));
    }

    void DoomController::tic(bool update) {

        if (this->doomRunning) {

            if (this->isTicPossible()) {
                this->mapLastTic = this->gameState->MAP_TIC + 1;
                if (update) this->MQDoom->send(MSG_CODE_TIC_AND_UPDATE);
                else this->MQDoom->send(MSG_CODE_TIC);
                this->waitForDoomWork();
            }
        } else throw ViZDoomIsNotRunningException();
    }

    void DoomController::tics(unsigned int tics, bool update) {

        if (this->allowDoomInput && !this->runDoomAsync) {
            for (int i = 0; i < DELTA_BUTTON_COUNT; ++i) {
                this->input->BT_MAX_VALUE[i] = tics * this->_input->BT_MAX_VALUE[i];
            }
        }

        int ticsMade = 0;

        for (int i = 0; i < tics; ++i) {
            if (i == tics - 1) this->tic(update);
            else this->tic(false);

            ++ticsMade;

            if (!this->isTicPossible() && i != tics - 1) {
                this->MQDoom->send(MSG_CODE_UPDATE);
                this->waitForDoomWork();
                break;
            }
        }

        if (this->allowDoomInput && !this->runDoomAsync) {
            for (int i = BINARY_BUTTON_COUNT; i < BUTTON_COUNT; ++i) {
                this->input->BT_MAX_VALUE[i - BINARY_BUTTON_COUNT] = this->_input->BT_MAX_VALUE[i -
                                                                                                BINARY_BUTTON_COUNT];
                this->input->BT[i] = this->input->BT[i] / ticsMade;
            }
        }
    }

    void DoomController::restartMap(std::string demoPath) {
        this->setMap(this->map, demoPath);
    }

    void DoomController::respawnPlayer() {

        if (this->doomRunning && !this->mapChanging && !this->gameState->MAP_END && this->gameState->PLAYER_DEAD) {
            if (this->gameState->GAME_MULTIPLAYER) {

                bool useAvailable = this->input->BT_AVAILABLE[USE];
                this->input->BT_AVAILABLE[USE] = true;

                do {
                    this->sendCommand(std::string("+use"));

                    this->MQDoom->send(MSG_CODE_TIC);
                    this->waitForDoomWork();

                } while (!this->gameState->MAP_END && this->gameState->PLAYER_DEAD);

                this->sendCommand(std::string("-use"));
                this->MQDoom->send(MSG_CODE_UPDATE);
                this->waitForDoomWork();

                this->input->BT_AVAILABLE[USE] = useAvailable;
                this->mapLastTic = this->gameState->MAP_TIC;

            } else this->restartMap();
        }
    }

    void DoomController::sendCommand(std::string command) {
        if (this->doomRunning && this->MQDoom && command.length() <= MQ_MAX_CMD_LEN)
            this->MQDoom->send(MSG_CODE_COMMAND, command.c_str());
    }

    void DoomController::addCustomArg(std::string arg) {
        this->customArgs.push_back(arg);
    }

    void DoomController::clearCustomArgs() {
        this->customArgs.clear();
    }

    bool DoomController::isDoomRunning() { return this->doomRunning; }

    std::string DoomController::getMap() { return this->map; }

    void DoomController::setMap(std::string map, std::string demoPath) {
        this->map = map;
        this->demoPath = demoPath;

        if (this->doomRunning && !this->mapChanging) {

            if (this->gameState->DEMO_RECORDING) this->sendCommand("stop");


            if(this->gameState->GAME_MULTIPLAYER){
                this->setDoomSeed(this->getNextDoomSeed());
                if(this->gameState->GAME_SETTINGS_CONTROLLER) this->sendCommand(std::string("changemap ") + this->map);
            }
            else if(this->demoPath.length()){
                this->forceDoomSeed(this->getNextDoomSeed());
                this->sendCommand(std::string("recordmap ") + prepareFilePathCmd(this->demoPath) + " " + this->map);
            }
            else {
                this->forceDoomSeed(this->getNextDoomSeed());
                this->sendCommand(std::string("map ") + this->map);
            }

            if (map != this->map) this->mapRestartCount = 0;
            else ++this->mapRestartCount;

            this->mapChanging = true;

            this->resetButtons();
            int restartTics = 0;

            bool useAvailable = this->input->BT_AVAILABLE[USE];

            if (this->gameState->GAME_MULTIPLAYER) {
                this->input->BT_AVAILABLE[USE] = true;
                this->sendCommand(std::string("-use"));
            }

            do {
                ++restartTics;

                if (this->gameState->GAME_MULTIPLAYER) {
                    if (restartTics % 2) this->sendCommand(std::string("+use"));
                    else this->sendCommand(std::string("-use"));
                }

                this->MQDoom->send(MSG_CODE_TIC);
                this->waitForDoomWork();

                if (restartTics > 3 && !this->gameState->GAME_MULTIPLAYER) {
                    if (this->demoPath.length())
                        this->sendCommand(std::string("recordmap ") + this->demoPath + " " + this->map);
                    else this->sendCommand(std::string("map ") + this->map);
                    restartTics = 0;
                }

            } while (this->gameState->MAP_END || this->gameState->MAP_TIC > this->mapLastTic);

            if (this->gameState->GAME_MULTIPLAYER) {
                this->sendCommand(std::string("-use"));
                this->input->BT_AVAILABLE[USE] = useAvailable;
            }

            this->waitForDoomMapStartTime();

            this->sendCommand("viz_override_player 0");

            this->MQDoom->send(MSG_CODE_UPDATE);
            this->waitForDoomWork();

            this->mapLastTic = this->gameState->MAP_TIC;
            this->mapChanging = false;
        }
    }

    void DoomController::playDemo(std::string demoPath, int player) {
        if (this->doomRunning && !this->mapChanging) {

            if (this->gameState->DEMO_RECORDING) this->sendCommand("stop");

            this->sendCommand(std::string("playdemo ") + prepareLmpFilePath(demoPath));

            this->mapChanging = true;

            this->resetButtons();
            int restartTics = 0;

            do {
                ++restartTics;

                this->MQDoom->send(MSG_CODE_TIC);
                this->waitForDoomWork();

                if (restartTics > 3) {
                    this->sendCommand(std::string("playdemo ") + demoPath);
                    restartTics = 0;
                }

            } while (this->gameState->MAP_END || this->gameState->MAP_TIC > this->mapLastTic);

            this->waitForDoomMapStartTime();

            this->sendCommand(std::string("viz_override_player ") + b::lexical_cast<std::string>(player));

            this->MQDoom->send(MSG_CODE_UPDATE);
            this->waitForDoomWork();

            this->mapLastTic = this->gameState->MAP_TIC;
            this->mapChanging = false;
        }
    }


    /* Settings */
    /*----------------------------------------------------------------------------------------------------------------*/

    unsigned int DoomController::getTicrate() { return this->ticrate; }

    void DoomController::setTicrate(unsigned int ticrate) { this->ticrate = ticrate; }

    std::string DoomController::getExePath() { return this->exePath; }

    void DoomController::setExePath(std::string exePath) { if (!this->doomRunning) this->exePath = exePath; }

    std::string DoomController::getIwadPath() { return this->iwadPath; }

    void DoomController::setIwadPath(std::string iwadPath) { if (!this->doomRunning) this->iwadPath = iwadPath; }

    std::string DoomController::getFilePath() { return this->filePath; }

    void DoomController::setFilePath(std::string filePath) {
        this->filePath = filePath;
        if (this->doomRunning) {
            this->map = "file:" + prepareWadFilePath(this->filePath);
        }
    }

    std::string DoomController::getConfigPath() { return this->configPath; }

    void
    DoomController::setConfigPath(std::string configPath) { if (!this->doomRunning) this->configPath = configPath; }

    int DoomController::getSkill() { return this->skill; }

    void DoomController::setSkill(int skill) {
        if (skill > 5) skill = 5;
        else if (skill < 1) skill = 1;
        this->skill = skill;
        if (this->doomRunning) {
            this->sendCommand(std::string("skill set ") + b::lexical_cast<std::string>(this->skill - 1));
            //this->resetMap(); // needs map restart to take effect
        }
    }

    unsigned int DoomController::getDoomSeed() {
        if (this->doomRunning) return this->gameState->GAME_STATIC_SEED;
        else return this->doomSeed;
    }

    void DoomController::setDoomSeed(unsigned int seed) {
        this->doomStaticSeed = true;
        this->doomSeed = seed;
        if (this->doomRunning) {
            this->sendCommand(std::string("rngseed set ") + b::lexical_cast<std::string>(this->doomSeed));
        }
    }

    void DoomController::clearDoomSeed() {
        this->doomStaticSeed = false;
        this->doomSeed = 0;
        if (this->doomRunning) {
            this->sendCommand("rngseed clear");
        }
    }

    void DoomController::setInstanceSeed(unsigned int seed) {
        this->instanceSeed = seed;
        this->instanceRng.seed(seed);
    }

    unsigned int DoomController::getInstanceSeed() { return this->instanceSeed; }

    unsigned int DoomController::getMapStartTime() { return this->mapStartTime; }

    void DoomController::setMapStartTime(unsigned int tics) { this->mapStartTime = tics ? tics : 1; }

    unsigned int DoomController::getMapTimeout() { return this->mapTimeout; }

    void DoomController::setMapTimeout(unsigned int tics) { this->mapTimeout = tics; }

    bool DoomController::isMapFirstTic() {
        return this->doomRunning && this->gameState->MAP_TIC <= 1;
    }

    bool DoomController::isMapLastTic() {
        return this->doomRunning && this->mapTimeout > 0
               && this->gameState->MAP_TIC >= this->mapTimeout + this->mapStartTime;
    }

    bool DoomController::isMapEnded() {
        return this->doomRunning && this->gameState->MAP_END;
    }

    unsigned int DoomController::getMapLastTic() {
        return this->mapLastTic;
    }

    void DoomController::setNoConsole(bool console) {
        if (!this->doomRunning) this->noConsole = console;
    }

    void DoomController::setNoSound(bool sound) {
        if (!this->doomRunning) this->noSound = sound;
    }

    void DoomController::setScreenResolution(unsigned int width, unsigned int height) {
        if (!this->doomRunning) {
            this->screenWidth = width;
            this->screenHeight = height;
        }
    }

    /* Depth buffer */
    bool DoomController::isDepthBufferEnabled() {
        if (this->doomRunning) return this->gameState->DEPTH_BUFFER;
        else return depth;
    }

    void DoomController::setDepthBufferEnabled(bool depthBuffer) {
        this->depth = depthBuffer;
        if (this->doomRunning) {
            if (this->automap) this->sendCommand("viz_depth 1");
            else this->sendCommand("viz_depth 0");
        }
        this->updateSettings = true;
    }

    /* Labels */
    bool DoomController::isLabelsEnabled() {
        if (this->doomRunning) return this->gameState->LABELS;
        else return labels;
    }

    void DoomController::setLabelsEnabled(bool labels) {
        this->labels = labels;
        if (this->doomRunning) {
            if (this->automap) this->sendCommand("viz_labels 1");
            else this->sendCommand("viz_labels 0");
        }
    }

    /* Automap */
    bool DoomController::isAutomapEnabled() {
        if (this->doomRunning) return this->gameState->AUTOMAP;
        else return automap;
    }

    void DoomController::setAutomapEnabled(bool automap) {
        this->automap = automap;
        if (this->doomRunning) {
            if (this->automap) this->sendCommand("viz_automap 1");
            else this->sendCommand("viz_automap 0");
        }
    }

    void DoomController::setAutomapMode(AutomapMode mode) {
        this->amMode = mode;
        if (this->doomRunning) this->sendCommand("viz_automap_mode " + b::lexical_cast<std::string>(this->amMode));
    }

    void DoomController::setAutomapRotate(bool rotate) {
        this->amRotate = rotate;
        if (this->doomRunning) this->setRenderMode(this->getRenderModeValue());
    }

    void DoomController::setAutomapRenderTextures(bool textures) {
        this->amTextures = textures;
        if (this->doomRunning) this->setRenderMode(this->getRenderModeValue());
    }

    void DoomController::setScreenWidth(unsigned int width) {
        if (!this->doomRunning) this->screenWidth = width;
    }

    void DoomController::setScreenHeight(unsigned int height) {
        if (!this->doomRunning) this->screenHeight = height;
    }

    void DoomController::setScreenFormat(ScreenFormat format) {
        if (!this->doomRunning) {
            this->screenFormat = format;
            switch (format) {
                case CRCGCB:
                case RGB24:
                case CBCGCR:
                case BGR24:
                    this->screenChannels = 3;
                    break;
                case RGBA32:
                case ARGB32:
                case BGRA32:
                case ABGR32:
                    this->screenChannels = 4;
                    break;
                case GRAY8:
                case DOOM_256_COLORS8:
                    this->screenChannels = 1;
                    break;
                default:
                    this->screenChannels = 0;
            }

            switch (format) {
                case RGB24:
                case BGR24:
                    this->screenDepth = 24;
                    break;
                case RGBA32:
                case ARGB32:
                case BGRA32:
                case ABGR32:
                    this->screenDepth = 32;
                    break;
                case CRCGCB:
                case CBCGCR:
                case GRAY8:
                case DOOM_256_COLORS8:
                    this->screenDepth = 8;
                    break;
                default:
                    this->screenDepth = 0;
            }
        }
        if (this->doomRunning) {
            this->sendCommand("viz_screen_format " + b::lexical_cast<std::string>(this->screenFormat));
        }
    }

    void DoomController::setWindowHidden(bool windowHidden) {
        if (!this->doomRunning) this->windowHidden = windowHidden;
    }

    void DoomController::setNoXServer(bool noXServer) {
        if (!this->doomRunning) this->noXServer = noXServer;
    }

    void DoomController::setRenderHud(bool hud) {
        this->hud = hud;
        if (this->doomRunning) this->setRenderMode(this->getRenderModeValue());
    }

    void DoomController::setRenderMinimalHud(bool minHud) {
        this->minHud = minHud;
        if (this->doomRunning) this->setRenderMode(this->getRenderModeValue());
    }

    void DoomController::setRenderWeapon(bool weapon) {
        this->weapon = weapon;
        if (this->doomRunning) this->setRenderMode(this->getRenderModeValue());
    }

    void DoomController::setRenderCrosshair(bool crosshair) {
        this->crosshair = crosshair;
        if (this->doomRunning) this->setRenderMode(this->getRenderModeValue());
    }

    void DoomController::setRenderDecals(bool decals) {
        this->decals = decals;
        if (this->doomRunning) this->setRenderMode(this->getRenderModeValue());
    }

    void DoomController::setRenderParticles(bool particles) {
        this->particles = particles;
        if (this->doomRunning) this->setRenderMode(this->getRenderModeValue());
    }

    void DoomController::setRenderEffectsSprites(bool sprites) {
        this->sprites = sprites;
        if (this->doomRunning) this->setRenderMode(this->getRenderModeValue());
    }

    void DoomController::setRenderMessages(bool messages) {
        this->messages = messages;
        if (this->doomRunning) this->setRenderMode(this->getRenderModeValue());
    }

    void DoomController::setRenderCorpses(bool corpses){
        this->corpses = corpses;
        if (this->doomRunning) this->setRenderMode(this->getRenderModeValue());
    }

    void DoomController::setRenderScreenFlashes(bool flashes){
        this->flashes = flashes;
        if (this->doomRunning) this->setRenderMode(this->getRenderModeValue());
    }

    unsigned int DoomController::getScreenWidth() {
        if (this->doomRunning) return this->gameState->SCREEN_WIDTH;
        else return this->screenWidth;
    }

    unsigned int DoomController::getScreenHeight() {
        if (this->doomRunning) return this->gameState->SCREEN_HEIGHT;
        else return this->screenHeight;
    }

    unsigned int DoomController::getScreenChannels() { return this->screenChannels; }

    unsigned int DoomController::getScreenDepth() { return this->screenDepth; }

    size_t DoomController::getScreenPitch() {
        if (this->doomRunning) return this->gameState->SCREEN_PITCH;
        else return static_cast<size_t>(this->screenDepth / 8 * this->screenWidth);
    }

    ScreenFormat DoomController::getScreenFormat() {
        if (this->doomRunning) return static_cast<ScreenFormat>(this->gameState->SCREEN_FORMAT);
        else return this->screenFormat;
    }

    size_t DoomController::getScreenSize() {
        if (this->doomRunning) return (size_t) this->gameState->SCREEN_SIZE;
        else return (size_t) this->screenChannels * this->screenWidth * this->screenHeight;
    }

    int DoomController::getRenderModeValue() {
        int renderMode = 0;

        if(this->hud)           renderMode |= 1;
        if(this->minHud)        renderMode |= 2;
        if(this->weapon)        renderMode |= 4;
        if(this->crosshair)     renderMode |= 8;
        if(this->decals)        renderMode |= 16;
        if(this->particles)     renderMode |= 32;
        if(this->sprites)       renderMode |= 64;
        if(this->messages)      renderMode |= 128;
        if(this->amRotate)      renderMode |= 256;
        if(this->amTextures)    renderMode |= 512;
        if(this->corpses)       renderMode |= 1024;
        if(this->flashes)       renderMode |= 2048;

        return renderMode;
    }

    void DoomController::setRenderMode(int value) {
        this->sendCommand("viz_render_mode " + b::lexical_cast<std::string>(this->getRenderModeValue()));
    }

    /* SM setters & getters */
    /*----------------------------------------------------------------------------------------------------------------*/

    uint8_t *const DoomController::getScreenBuffer() { return this->screenBuffer; }

    uint8_t *const DoomController::getDepthBuffer() { return this->depthBuffer; }

    uint8_t *const DoomController::getLabelsBuffer() { return this->labelsBuffer; }

    uint8_t *const DoomController::getAutomapBuffer() { return this->automapBuffer; }

    SMInputState *const DoomController::getInput() { return this->input; }

    SMGameState *const DoomController::getGameState() { return this->gameState; }

    int DoomController::getButtonState(Button button) {
        if (this->doomRunning) return this->input->BT[button];
        else return 0;
    }

    void DoomController::setButtonState(Button button, int state) {
        if (button < BUTTON_COUNT && button >= 0 && this->doomRunning)
            this->input->BT[button] = state;
    }

    void DoomController::toggleButtonState(Button button) {
        if (button < BUTTON_COUNT && button >= 0 && this->doomRunning)
            this->input->BT[button] = !this->input->BT[button];
    }

    bool DoomController::isButtonAvailable(Button button) {
        if (this->doomRunning) return this->input->BT_AVAILABLE[button];
        else return this->_input->BT_AVAILABLE[button];
    }

    void DoomController::setButtonAvailable(Button button, bool allow) {
        if (button < BUTTON_COUNT && button >= 0) {
            if (this->doomRunning) this->input->BT_AVAILABLE[button] = allow;
            this->_input->BT_AVAILABLE[button] = allow;
        }
    }

    void DoomController::resetButtons() {
        if (this->doomRunning)
            for (int i = 0; i < BUTTON_COUNT; ++i)
                this->input->BT[i] = 0;
    }

    void DoomController::disableAllButtons() {
        for (int i = 0; i < BUTTON_COUNT; ++i) {
            if (this->doomRunning) this->input->BT_AVAILABLE[i] = false;
            this->_input->BT_AVAILABLE[i] = false;
        }
    }

    void DoomController::availableAllButtons() {
        for (int i = 0; i < BUTTON_COUNT; ++i) {
            if (this->doomRunning) this->input->BT_AVAILABLE[i] = true;
            this->_input->BT_AVAILABLE[i] = true;
        }
    }

    void DoomController::setButtonMaxValue(Button button, unsigned int value) {
        if (button >= BINARY_BUTTON_COUNT) {
            if (this->doomRunning) this->input->BT_MAX_VALUE[button - BINARY_BUTTON_COUNT] = value;
            this->_input->BT_MAX_VALUE[button - BINARY_BUTTON_COUNT] = value;
        }
    }

    int DoomController::getButtonMaxValue(Button button) {
        if (button >= BINARY_BUTTON_COUNT) {
            if (this->doomRunning) return this->input->BT_MAX_VALUE[button - BINARY_BUTTON_COUNT];
            else return this->_input->BT_MAX_VALUE[button - BINARY_BUTTON_COUNT];
        } else return 1;
    }

    bool DoomController::isAllowDoomInput() { return this->allowDoomInput; }

    void DoomController::setAllowDoomInput(bool set) { if (!this->doomRunning) this->allowDoomInput = set; }

    bool DoomController::isRunDoomAsync() { return this->runDoomAsync; }

    void DoomController::setRunDoomAsync(bool set) { if (!this->doomRunning) this->runDoomAsync = set; }

    /* GameVariables getters */
    /*----------------------------------------------------------------------------------------------------------------*/


    double DoomController::getGameVariable(GameVariable var){

        switch (var) {
            case KILLCOUNT :
                return this->gameState->MAP_KILLCOUNT;
            case ITEMCOUNT :
                return this->gameState->MAP_ITEMCOUNT;
            case SECRETCOUNT :
                return this->gameState->MAP_SECRETCOUNT;
            case FRAGCOUNT:
                return this->gameState->PLAYER_FRAGCOUNT;
            case DEATHCOUNT:
                return this->gameState->PLAYER_DEATHCOUNT;
            case HEALTH :
                return this->gameState->PLAYER_HEALTH;
            case ARMOR :
                return this->gameState->PLAYER_ARMOR;
            case DEAD :
                return this->gameState->PLAYER_DEAD;
            case ON_GROUND :
                return static_cast<double>(this->gameState->PLAYER_ON_GROUND);
            case ATTACK_READY :
                return static_cast<double>(this->gameState->PLAYER_ATTACK_READY);
            case ALTATTACK_READY :
                return static_cast<double>(this->gameState->PLAYER_ALTATTACK_READY);
            case SELECTED_WEAPON :
                return this->gameState->PLAYER_SELECTED_WEAPON;
            case SELECTED_WEAPON_AMMO :
                return this->gameState->PLAYER_SELECTED_WEAPON_AMMO;
            case PLAYER_NUMBER:
                return static_cast<double>(this->gameState->PLAYER_NUMBER);
            case PLAYER_COUNT:
                return static_cast<double>(this->gameState->PLAYER_COUNT);
        }

        if (var >= AMMO0 && var <= AMMO9) {
            return this->gameState->PLAYER_AMMO[var - AMMO0];
        } else if (var >= WEAPON0 && var <= WEAPON9) {
            return this->gameState->PLAYER_WEAPON[var - WEAPON0];
        }
        else if(var >= POSITION_X && var <= POSITION_Z){
            return this->gameState->PLAYER_POSITION[var - POSITION_X];
        }
        else if(var >= USER1 && var <= USER30){

            return this->gameState->MAP_USER_VARS[var - USER1];
        } else if (var >= PLAYER1_FRAGCOUNT && var <= PLAYER8_FRAGCOUNT) {
            return this->gameState->PLAYER_N_FRAGCOUNT[var - PLAYER1_FRAGCOUNT];
        }

        return 0;
    }

    unsigned int DoomController::getGameTic() { return this->gameState->GAME_TIC; }

    bool DoomController::isMultiplayerGame() { return this->gameState->GAME_MULTIPLAYER; }

    bool DoomController::isNetGame() { return this->gameState->GAME_NETGAME; }

    unsigned int DoomController::getMapTic() { return this->gameState->MAP_TIC; }

    int DoomController::getMapReward() { return this->gameState->MAP_REWARD; }

    bool DoomController::isPlayerDead() { return this->gameState->PLAYER_DEAD; }

    /* Protected and private functions */
    /*----------------------------------------------------------------------------------------------------------------*/

    void DoomController::generateInstanceId() {
        std::string chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";
        this->instanceId = "";

        br::uniform_int_distribution<> charDist(0, static_cast<int>(chars.length() - 1));
        br::mt19937 rng;
        rng.seed((unsigned int) bc::high_resolution_clock::now().time_since_epoch().count());

        for (int i = 0; i < INSTANCE_ID_LENGHT; ++i) {
            this->instanceId += chars[charDist(rng)];
        }
    }

    unsigned int DoomController::getNextDoomSeed() {
        br::uniform_int_distribution<> mapSeedDist(0, UINT_MAX);
        return static_cast<unsigned int>(mapSeedDist(this->instanceRng));
    }

    void DoomController::forceDoomSeed(unsigned int seed) {
        this->doomStaticSeed = true;
        this->doomSeed = seed;
        if (this->doomRunning) {
            this->sendCommand(std::string("viz_set_seed ") + b::lexical_cast<std::string>(this->doomSeed));
        }
    }

    /* Signals */
    /*----------------------------------------------------------------------------------------------------------------*/

    void DoomController::handleSignals() {
        this->ioService = new ba::io_service();
        ba::signal_set signals(*this->ioService, SIGINT, SIGABRT, SIGTERM);
        signals.async_wait(b::bind(signalHandler, b::ref(signals), this, _1, _2));

        this->ioService->run();
    }

    void DoomController::signalHandler(ba::signal_set &signal, DoomController *controller, const bs::error_code &error,
                                       int sigNumber) {
        controller->intSignal(sigNumber);
    }

    void DoomController::intSignal(int sigNumber) {
        this->MQDoom->send(MSG_CODE_CLOSE);
        this->MQController->send(static_cast<uint8_t >(MSG_CODE_SIG + sigNumber));
    }

    /* Flow */
    /*----------------------------------------------------------------------------------------------------------------*/

    bool DoomController::receiveMQMsg() {
        bool done = false;

        Message msg = this->MQController->receive();
        switch (msg.code) {
            case MSG_CODE_DOOM_DONE :
                done = true;
                break;

            case MSG_CODE_DOOM_CLOSE :
            case MSG_CODE_DOOM_PROCESS_EXIT :
                this->close();
                throw ViZDoomUnexpectedExitException();

            case MSG_CODE_DOOM_ERROR :
                this->close();
                throw ViZDoomErrorException(std::string(msg.command));

            case MSG_CODE_SIGINT :
                this->close();
                throw SignalException("SIGINT");

            case MSG_CODE_SIGABRT :
                this->close();
                throw SignalException("SIGABRT");

            case MSG_CODE_SIGTERM :
                this->close();
                throw SignalException("SIGTERM");

            default:
                this->close();
                throw MessageQueueException("Unknown message code. Possible ViZDoom version mismatch.");
        }

        return done;
    }

    void DoomController::waitForDoomStart() {
        this->doomWorking = true;
        this->doomRunning = this->receiveMQMsg();
        this->doomWorking = false;
    }

    void DoomController::waitForDoomWork() {
        if (doomRunning) {
            this->doomWorking = true;

            bool done;
            do {
                done = this->receiveMQMsg();
            } while (!done);

            this->doomWorking = false;
        } else throw ViZDoomIsNotRunningException();
    }

    void DoomController::waitForDoomMapStartTime() {
        while (this->gameState->MAP_TIC < this->mapStartTime) {
            this->MQDoom->send(MSG_CODE_TIC);
            this->waitForDoomWork();
        }
    }

    /* Init */
    /*----------------------------------------------------------------------------------------------------------------*/

    void DoomController::createDoomArgs() {
        this->doomArgs.clear();

        // exe
        if (this->exePath.length() == 0){
            std::string workingExePath = "./vizdoom";
            std::string sharedExePath = getThisSharedObjectPath() + "/vizdoom";

            #ifdef OS_WIN
                workingExePath += ".exe";
                sharedExePath += ".exe";
            #endif

            if (fileExists(workingExePath)) this->exePath = workingExePath;
            else if (fileExists(sharedExePath)) this->exePath = sharedExePath;
            else throw FileDoesNotExistException(workingExePath + " | " + sharedExePath);
        }

        this->doomArgs.push_back(prepareExeFilePath(this->exePath));

        // main wad
        if (this->iwadPath.length() == 0) {
            std::string workingDoom2Path = "./doom2.wad";
            std::string workingFreedoom2Path = "./freedoom2.wad";
            std::string sharedDoom2Path = getThisSharedObjectPath() + "/doom2.wad";
            std::string sharedFreedoom2Path = getThisSharedObjectPath() + "/freedoom2.wad";

            if (fileExists(workingDoom2Path)) this->iwadPath = workingDoom2Path;
            else if (fileExists(sharedDoom2Path)) this->iwadPath = sharedDoom2Path;
            else if (fileExists(workingFreedoom2Path)) this->iwadPath = workingFreedoom2Path;
            else if (fileExists(sharedFreedoom2Path)) this->iwadPath = sharedFreedoom2Path;
            else throw FileDoesNotExistException(workingDoom2Path
                                                 + " | " + workingFreedoom2Path
                                                 + " | " + sharedDoom2Path
                                                 + " | " + sharedFreedoom2Path);
        }

        this->doomArgs.push_back("-iwad");
        this->doomArgs.push_back(prepareWadFilePath(this->iwadPath));

        // scenario wad
        if (this->filePath.length() != 0) {
            this->doomArgs.push_back("-file");
            this->doomArgs.push_back(prepareWadFilePath(this->filePath));
        }

        // config
        this->doomArgs.push_back("-config");
        if (this->configPath.length() != 0) this->doomArgs.push_back(prepareFilePathArg(this->configPath));
        else this->doomArgs.push_back("_vizdoom.ini");

        // map
        this->doomArgs.push_back("+map");
        if (this->map.length() > 0) this->doomArgs.push_back(this->map);
        else this->doomArgs.push_back("map01");

        // demo recording
        //if (this->demoPath.length() != 0){
        //    this->doomArgs.push_back("-record");
        //    this->doomArgs.push_back(this->demoPath);
        //    this->doomRecordingMap = true;
        //}
        //else this->doomRecordingMap = false;

        // skill
        this->doomArgs.push_back("-skill");
        this->doomArgs.push_back(b::lexical_cast<std::string>(this->skill));

        // resolution and aspect ratio
        this->doomArgs.push_back("-width");
        this->doomArgs.push_back(b::lexical_cast<std::string>(this->screenWidth));

        this->doomArgs.push_back("-height");
        this->doomArgs.push_back(b::lexical_cast<std::string>(this->screenHeight));

        float ratio = this->screenWidth / this->screenHeight;

        this->doomArgs.push_back("+vid_aspect");
        if (ratio == 16.0 / 9.0) this->doomArgs.push_back("1");
        else if (ratio == 16.0 / 10.0) this->doomArgs.push_back("2");
        else if (ratio == 4.0 / 3.0) this->doomArgs.push_back("3");
        else if (ratio == 5.0 / 4.0) this->doomArgs.push_back("4");
        else this->doomArgs.push_back("0");

        // window mode
        this->doomArgs.push_back("+fullscreen");
        this->doomArgs.push_back("0");

        // vizdoom
        this->doomArgs.push_back("+viz_controlled");
        this->doomArgs.push_back("1");

        this->doomArgs.push_back("+viz_instance_id");
        this->doomArgs.push_back(this->instanceId);

        // mode
        if (this->runDoomAsync) {
            this->doomArgs.push_back("+viz_async");
            this->doomArgs.push_back("1");
        }

        if (this->allowDoomInput) {
            this->doomArgs.push_back("+viz_allow_input");
            this->doomArgs.push_back("1");

            //allow mouse
            this->doomArgs.push_back("+use_mouse");
            this->doomArgs.push_back("1");

            #ifdef OS_WIN
                // Fix for problem with delta buttons' last values on Windows.
                this->doomArgs.push_back("+in_mouse");
                this->doomArgs.push_back("2");
            #endif

        } else {
            //disable mouse
            this->doomArgs.push_back("+use_mouse");
            this->doomArgs.push_back("0");
        }

        // seed
        if (this->doomStaticSeed) {
            this->forceDoomSeed(this->getNextDoomSeed());
            this->doomArgs.push_back("-rngseed");
            this->doomArgs.push_back(b::lexical_cast<std::string>(this->doomSeed));
        }

        // depth duffer
        if (this->depth) {
            this->doomArgs.push_back("+viz_depth");
            this->doomArgs.push_back("1");
        }

        // labels
        if (this->labels) {
            this->doomArgs.push_back("+viz_labels");
            this->doomArgs.push_back("1");
        }

        // automap
        if (this->automap) {
            this->doomArgs.push_back("+viz_automap");
            this->doomArgs.push_back("1");

            this->doomArgs.push_back("+viz_automap_mode");
            this->doomArgs.push_back(b::lexical_cast<std::string>(this->amMode));
        }

        // render mode
        this->doomArgs.push_back("+viz_render_mode");
        this->doomArgs.push_back(b::lexical_cast<std::string>(this->getRenderModeValue()));

        if (this->noConsole) {
            this->doomArgs.push_back("+viz_noconsole");
            this->doomArgs.push_back("1");
        }

        this->doomArgs.push_back("+viz_screen_format");
        this->doomArgs.push_back(b::lexical_cast<std::string>(this->screenFormat));

        this->doomArgs.push_back("+viz_window_hidden");
        if (this->windowHidden) this->doomArgs.push_back("1");
        else this->doomArgs.push_back("0");

        #ifdef OS_LINUX
            this->doomArgs.push_back("+viz_noxserver");
            if (this->noXServer) this->doomArgs.push_back("1");
            else this->doomArgs.push_back("0");
        #endif

        // idle/joy
        this->doomArgs.push_back("-noidle");
        this->doomArgs.push_back("-nojoy");

        // sound
        if (this->noSound) {
            this->doomArgs.push_back("-nosound");
            this->doomArgs.push_back("+viz_nosound");
            this->doomArgs.push_back("1");
        }

        if (this->ticrate != DEFAULT_TICRATE) {
            this->doomArgs.push_back("-ticrate");
            this->doomArgs.push_back(b::lexical_cast<std::string>(this->ticrate));
        }

        //custom args
        for (int i = 0; i < this->customArgs.size(); ++i) {
            this->doomArgs.push_back(customArgs[i]);
        }
    }

    void DoomController::launchDoom() {
        try {
            bpr::child doomProcess = bpr::execute(bpri::set_args(this->doomArgs), bpri::inherit_env());
            bpr::wait_for_exit(doomProcess);
        }
        catch (...) {
            this->MQController->send(MSG_CODE_DOOM_ERROR, "Unexpected ViZDoom instance crash.");
        }
        this->MQController->send(MSG_CODE_DOOM_PROCESS_EXIT);
    }
}
