/*
 Copyright (C) 2016 by Wojciech Jaśkowski, Michał Kempka, Grzegorz Runc, Jakub Toczek, Marek Wydmuch
 Copyright (C) 2017 - 2022 by Marek Wydmuch, Michał Kempka, Wojciech Jaśkowski, and the respective contributors
 Copyright (C) 2023 - 2025 by Marek Wydmuch, Farama Foundation, and the respective contributors

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

#include <boost/algorithm/string.hpp>
#include <boost/chrono.hpp>
#include <boost/lexical_cast.hpp>


namespace vizdoom {

    namespace bal       = boost::algorithm;
    namespace bc        = boost::chrono;
    namespace bfs       = boost::filesystem;


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
        this->doomThread = nullptr;

        /* Flow control */
        this->doomRunning = false;
        this->doomWorking = false;

        this->mapStartTime = 1;
        this->mapTimeout = 0;
        this->mapRestartCount = 0;
        this->mapChanging = false;
        this->mapLastTic = 1;

        // TODO move default settings to separate file (ViZDoomConsts.h perhaps?)
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

        this->objects = false;
        this->sectors = false;

        this->softSoundAudio = false;
        this->audioSamplingFreq = 44100;
        this->audioSamplesPerTic = this->audioSamplingFreq / int(DEFAULT_TICRATE);
        this->audioBufferSizeInTics = 1;

        this->notifications = false;
        this->notificationsBufferSizeInTics = 1;

        this->hud = false;
        this->minHud = false;
        this->weapon = true;
        this->crosshair = false;
        this->decals = true;
        this->particles = true;
        this->sprites = true;
        this->messages = false;
        this->corpses = true;
        this->renderAll = false;

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
                this->audioBuffer = this->SM->getAudioBuffer();
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
        try{
            if (this->doomRunning) {
                // Try to stop demo recording
                if (this->gameState->DEMO_RECORDING){
                    this->sendCommand("stop");
                    this->MQDoom->send(MSG_CODE_TIC);
                    this->waitForDoomWork();
                }

                this->doomRunning = false;
                this->doomWorking = false;
                
                // Try to close Doom gently
                this->MQDoom->send(MSG_CODE_CLOSE);
            }

            #ifdef OS_POSIX
            // If the Doom thread is still running, kill the engine process
            if (this->doomThread && !this->doomThread->joinable()) {
                if(0 == kill(this->doomProcessPid, 0)){
                    bpr::child doomProcess(this->doomProcessPid);
                    bpr::terminate(doomProcess);
                }
            }
            #endif

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
        }
        catch (...) { throw; }

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
        return !(this->isMapEnded() || this->isMapTimeoutReached());
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

        for (unsigned int i = 0; i < tics; ++i) {
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

        if (this->doomRunning && !this->mapChanging) {
            if (this->gameState->GAME_MULTIPLAYER && this->gameState->PLAYER_DEAD && isTicPossible()) {

                bool useAvailable = this->input->BT_AVAILABLE[USE];
                this->input->BT_AVAILABLE[USE] = true;

                do {
                    this->sendCommand(std::string("+use"));

                    this->MQDoom->send(MSG_CODE_TIC);
                    this->waitForDoomWork();

                    if(!isTicPossible()) return;
                } while (this->gameState->PLAYER_DEAD);

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

    std::vector<std::string> DoomController::getCustomArgs() {
        return this->customArgs;
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

            // Workaround for some problems
            this->sendCommand(std::string("map ") + this->map);
            this->MQDoom->send(MSG_CODE_TIC);
            this->waitForDoomWork();

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

    void DoomController::saveGame(std::string filePath){
        if (this->doomRunning && !this->mapChanging) {
            this->sendCommand(std::string("save ") + filePath);
            this->tics(2, false); // Saving takes 2 tics
        }
    }

    void DoomController::loadGame(std::string filePath){
        if (this->doomRunning && !this->mapChanging) {
            this->sendCommand(std::string("load ") + filePath);
            this->tic(true); // Loading takes 1 tic and needs update
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
        // Check if the map has ended or the player is dead (in single player mode)
        return (!this->gameState->GAME_MULTIPLAYER && this->gameState->PLAYER_DEAD) || (this->gameState->MAP_END);
    }

    bool DoomController::isMapTimeoutReached() {
        // Check if internal or user-defined map timeout has been reached
        return (this->mapTimeout > 0 && this->mapTimeout + this->mapStartTime <= this->gameState->MAP_TIC) 
                || (this->gameState->MAP_TICLIMIT > 0 && this->gameState->MAP_TICLIMIT <= this->gameState->MAP_TIC);
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

    bool DoomController::getNoSound() const {
        return this->noSound;
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
        if (!this->doomRunning) this->depth = depthBuffer;
    }

    /* Labels buffer */
    bool DoomController::isLabelsEnabled() {
        if (this->doomRunning) return this->gameState->LABELS;
        else return labels;
    }

    void DoomController::setLabelsEnabled(bool labels) {
        if (!this->doomRunning) this->labels = labels;
    }

    /* Automap buffer */
    bool DoomController::isAutomapEnabled() {
        if (this->doomRunning) return this->gameState->AUTOMAP;
        else return automap;
    }

    void DoomController::setAutomapEnabled(bool automap) {
        if (!this->doomRunning) this->automap = automap;
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

    /* Objects (actors) and map state */
    bool DoomController::isObjectsEnabled() {
        if (this->doomRunning) return this->gameState->OBJECTS;
        else return objects;
    }

    void DoomController::setObjectsEnabled(bool objects) {
        if (!this->doomRunning) this->objects = objects;
    }

    bool DoomController::isSectorsEnabled() {
        if (this->doomRunning) return this->gameState->SECTORS;
        else return sectors;
    }

    void DoomController::setSectorsEnabled(bool sectors) {
        if (!this->doomRunning) this->sectors = sectors;
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

    void DoomController::setRenderAllFrames(bool allFrames){
        this->renderAll = allFrames;
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

    bool DoomController::isAudioBufferEnabled() const{
        if (this->doomRunning) return this->gameState->AUDIO_BUFFER;
        else return this->softSoundAudio;
    }

    void DoomController::setAudioBufferEnabled(bool audioBuffer){
        if (!this->doomRunning) this->softSoundAudio = audioBuffer;
    }

    int DoomController::getAudioSamplingFreq() const {
        return this->audioSamplingFreq;
    }

    void DoomController::setAudioSamplingFreq(int freq) {
        if (!this->doomRunning){ 
            this->audioSamplingFreq = freq; 
            this->audioSamplesPerTic = freq / int(DEFAULT_TICRATE);
        }
    }

    int DoomController::getAudioSamplesPerTic() {
        return this->audioSamplesPerTic;
    }

    int DoomController::getAudioBufferSize() const {
        return this->audioBufferSizeInTics;
    }

    void DoomController::setAudioBufferSize(int tics) {
        if (!this->doomRunning) this->audioBufferSizeInTics = tics;
    }

    bool DoomController::isNotificationsEnabled() const {
        if (this->doomRunning) return this->gameState->NOTIFICATIONS;
        else return this->notifications;
    }

    void DoomController::setNotificationsEnabled(bool notifications) {
        if (!this->doomRunning) this->notifications = notifications;
    }

    int DoomController::getNotificationsBufferSize() const {
        return this->notificationsBufferSizeInTics;
    }

    void DoomController::setNotificationsBufferSize(int tics) {
        if (!this->doomRunning) this->notificationsBufferSizeInTics = tics;
    }

    /* SM setters & getters */
    /*----------------------------------------------------------------------------------------------------------------*/

    uint8_t *const DoomController::getScreenBuffer() { return this->screenBuffer; }

    int16_t *const DoomController::getAudioBuffer() { return this->audioBuffer; }

    uint8_t *const DoomController::getDepthBuffer() { return this->depthBuffer; }

    uint8_t *const DoomController::getLabelsBuffer() { return this->labelsBuffer; }

    uint8_t *const DoomController::getAutomapBuffer() { return this->automapBuffer; }

    SMInputState *const DoomController::getInput() { return this->input; }

    SMGameState *const DoomController::getGameState() { return this->gameState; }

    double DoomController::getButtonState(Button button) {
        if (this->doomRunning){
            if(this->isReplaying()) return this->input->CMD_BT[button];
            else return this->input->BT[button];
        }
        else return 0;
    }

    void DoomController::setButtonState(Button button, double state) {
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

    void DoomController::setButtonMaxValue(Button button, double value) {
        if (button >= BINARY_BUTTON_COUNT) {
            if (this->doomRunning) this->input->BT_MAX_VALUE[button - BINARY_BUTTON_COUNT] = value;
            this->_input->BT_MAX_VALUE[button - BINARY_BUTTON_COUNT] = value;
        }
    }

    double DoomController::getButtonMaxValue(Button button) {
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
            case KILLCOUNT:
                return this->gameState->MAP_KILLCOUNT;
            case ITEMCOUNT:
                return this->gameState->MAP_ITEMCOUNT;
            case SECRETCOUNT:
                return this->gameState->MAP_SECRETCOUNT;
            case FRAGCOUNT:
                return this->gameState->PLAYER_FRAGCOUNT;
            case DEATHCOUNT:
                return this->gameState->PLAYER_DEATHCOUNT;
            case HITCOUNT:
                return this->gameState->PLAYER_HITCOUNT;
            case HITS_TAKEN:
                return this->gameState->PLAYER_HITS_TAKEN;
            case DAMAGECOUNT:
                return this->gameState->PLAYER_DAMAGECOUNT;
            case DAMAGE_TAKEN:
                return this->gameState->PLAYER_DAMAGE_TAKEN;
            case HEALTH:
                return this->gameState->PLAYER_HEALTH;
            case ARMOR:
                return this->gameState->PLAYER_ARMOR;
            case DEAD:
                return this->gameState->PLAYER_DEAD;
            case ON_GROUND:
                return static_cast<double>(this->gameState->PLAYER_ON_GROUND);
            case ATTACK_READY:
                return static_cast<double>(this->gameState->PLAYER_ATTACK_READY);
            case ALTATTACK_READY:
                return static_cast<double>(this->gameState->PLAYER_ALTATTACK_READY);
            case SELECTED_WEAPON:
                return this->gameState->PLAYER_SELECTED_WEAPON;
            case SELECTED_WEAPON_AMMO:
                return this->gameState->PLAYER_SELECTED_WEAPON_AMMO;
            case PLAYER_NUMBER:
                return static_cast<double>(this->gameState->PLAYER_NUMBER);
            case PLAYER_COUNT:
                return static_cast<double>(this->gameState->PLAYER_COUNT);
        }

        if (var >= AMMO0 && var <= AMMO9)
            return this->gameState->PLAYER_AMMO[var - AMMO0];
        else if (var >= WEAPON0 && var <= WEAPON9)
            return this->gameState->PLAYER_WEAPON[var - WEAPON0];
        else if(var >= POSITION_X && var <= VELOCITY_Z)
            return this->gameState->PLAYER_MOVEMENT[var - POSITION_X];
        else if(var >= CAMERA_POSITION_X && var <= CAMERA_FOV)
            return this->gameState->CAMERA[var - CAMERA_POSITION_X];
        else if(var >= USER1 && var <= USER60)
            return this->gameState->MAP_USER_VARS[var - USER1];
        else if (var >= PLAYER1_FRAGCOUNT && var <= PLAYER16_FRAGCOUNT)
            return this->gameState->PLAYER_N_FRAGCOUNT[var - PLAYER1_FRAGCOUNT];

        return 0;
    }

    unsigned int DoomController::getGameTic() { return this->gameState->GAME_TIC; }

    bool DoomController::isMultiplayerGame() { return this->gameState->GAME_MULTIPLAYER; }

    bool DoomController::isNetGame() { return this->gameState->GAME_NETGAME; }

    bool DoomController::isRecording() { return this->gameState->DEMO_RECORDING; }

    bool DoomController::isReplaying() { return this->gameState->DEMO_PLAYBACK; }

    bool DoomController::isOpenALSoundInitialized() { return this->gameState->OPENAL_SOUND; }

    unsigned int DoomController::getMapTic() { return this->gameState->MAP_TIC; }

    int DoomController::getMapReward() { return this->gameState->MAP_REWARD; }

    int DoomController::getKillCount() { return this->gameState->PLAYER_KILLCOUNT; }

    int DoomController::getItemCount() { return this->gameState->PLAYER_ITEMCOUNT; }

    int DoomController::getSecretCount() { return this->gameState->PLAYER_SECRETCOUNT; }

    int DoomController::getFragCount() { return this->gameState->PLAYER_FRAGCOUNT; }

    int DoomController::getHitCount() { return this->gameState->PLAYER_HITCOUNT; }
    
    int DoomController::getHitsTaken() { return this->gameState->PLAYER_HITS_TAKEN; }
    
    int DoomController::getDamageCount() { return this->gameState->PLAYER_DAMAGECOUNT; }

    int DoomController::getDamageTaken() { return this->gameState->PLAYER_DAMAGE_TAKEN; }

    int DoomController::getHealth() { return this->gameState->PLAYER_HEALTH; }

    int DoomController::getArmor() { return this->gameState->PLAYER_ARMOR; }

    bool DoomController::isPlayerDead() { return this->gameState->PLAYER_DEAD; }

    int DoomController::getPlayerCount() { return this->gameState->PLAYER_COUNT; }

    bool DoomController::isPlayerInGame(unsigned int playerNumber){
        return playerNumber < MAX_PLAYERS ? this->gameState->PLAYER_N_IN_GAME[playerNumber] : false;
    }

    int DoomController::getPlayerFrags(unsigned int playerNumber){
        return playerNumber < MAX_PLAYERS ? this->gameState->PLAYER_N_FRAGCOUNT[playerNumber] : 0;
    }

    std::string DoomController::getPlayerName(unsigned int playerNumber){
        return playerNumber < MAX_PLAYERS ? std::string(this->gameState->PLAYER_N_NAME[playerNumber]) : "";
    }

    bool DoomController::isPlayerAfk(unsigned int playerNumber){
        return playerNumber < MAX_PLAYERS ? this->gameState->PLAYER_N_AFK[playerNumber] : false;
    }

    unsigned int DoomController::getPlayerLastActionTic(unsigned int playerNumber){
        return playerNumber < MAX_PLAYERS ? this->gameState->PLAYER_N_LAST_ACTION_TIC[playerNumber] : 0;
    }

    unsigned int DoomController::getPlayerLastKillTic(unsigned int playerNumber){
        return playerNumber < MAX_PLAYERS ? this->gameState->PLAYER_N_LAST_KILL_TIC[playerNumber] : 0;
    }


    /* Protected and private functions */
    /*----------------------------------------------------------------------------------------------------------------*/

    void DoomController::generateInstanceId() {
        std::string chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";
        this->instanceId = "";

        br::uniform_int_distribution<size_t> charDist(0, static_cast<size_t>(chars.length() - 1));
        br::mt19937 rng;
        rng.seed((unsigned int) bc::high_resolution_clock::now().time_since_epoch().count());

        for (int i = 0; i < INSTANCE_ID_LENGTH; ++i) {
            this->instanceId += chars[charDist(rng)];
        }
    }

    unsigned int DoomController::getNextDoomSeed(){
        br::uniform_int_distribution<unsigned int> mapSeedDist(0, UINT_MAX);
        return mapSeedDist(this->instanceRng);
    }

    void DoomController::forceDoomSeed(unsigned int seed) {
        this->doomStaticSeed = true;
        this->doomSeed = seed;
        if (this->doomRunning) {
            this->sendCommand(std::string("viz_set_seed ") + b::lexical_cast<std::string>(this->doomSeed));
        }
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
            std::string workingDoomPath = "./doom.wad";
            std::string workingFreedoom2Path = "./freedoom2.wad";
            std::string workingFreedoomPath = "./freedoom.wad";
            std::string sharedDoom2Path = getThisSharedObjectPath() + "/doom2.wad";
            std::string sharedDoomPath = getThisSharedObjectPath() + "/doom.wad";
            std::string sharedFreedoom2Path = getThisSharedObjectPath() + "/freedoom2.wad";
            std::string sharedFreedoomPath = getThisSharedObjectPath() + "/freedoom.wad";

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

        //this->doomArgs.push_back("+vid_defwidth");
        //this->doomArgs.push_back(b::lexical_cast<std::string>(this->screenWidth));
        //this->doomArgs.push_back("+vid_defheight");
        //this->doomArgs.push_back(b::lexical_cast<std::string>(this->screenHeight));

        float ratio = static_cast<float>(this->screenWidth) / this->screenHeight;

        this->doomArgs.push_back("+vid_aspect");
        if (ratio == 16.0f / 9.0f) this->doomArgs.push_back("1");
        else if (ratio == 16.0f / 10.0f) this->doomArgs.push_back("2");
        else if (ratio == 4.0f / 3.0f) this->doomArgs.push_back("3");
        else if (ratio == 5.0f / 4.0f) this->doomArgs.push_back("4");
        else this->doomArgs.push_back("0");

        // window mode
        this->doomArgs.push_back("+fullscreen");
        this->doomArgs.push_back("0");

        #ifdef OS_WIN
            // Output to stdout on Windows
            this->doomArgs.push_back("-stdout");
        #endif

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

        // labels buffer
        if (this->labels) {
            this->doomArgs.push_back("+viz_labels");
            this->doomArgs.push_back("1");
        }

        // automap buffer
        if (this->automap) {
            this->doomArgs.push_back("+viz_automap");
            this->doomArgs.push_back("1");

            this->doomArgs.push_back("+viz_automap_mode");
            this->doomArgs.push_back(b::lexical_cast<std::string>(this->amMode));
        }

        // objects (actors) and map state
        if (this->objects) {
            this->doomArgs.push_back("+viz_objects");
            this->doomArgs.push_back("1");
        }

        if (this->sectors) {
            this->doomArgs.push_back("+viz_sectors");
            this->doomArgs.push_back("1");
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


        if (this->windowHidden){
            this->doomArgs.push_back("+viz_window_hidden");
            this->doomArgs.push_back("1");

            #ifdef OS_POSIX
                if (this->noXServer){
                    this->doomArgs.push_back("+viz_noxserver");
                    this->doomArgs.push_back("1");
                }
            #endif
        }
        else{
            if(this->renderAll){
                this->doomArgs.push_back("+viz_render_all");
                this->doomArgs.push_back("1");
            }
        }

        // idle/joy
        this->doomArgs.push_back("-noidle");
        this->doomArgs.push_back("-nojoy");

        // audio buffer + sound
        if (this->softSoundAudio) {
            this->doomArgs.push_back("+viz_soft_audio");
            this->doomArgs.push_back("1");
            this->doomArgs.push_back("+snd_backend");
            this->doomArgs.push_back("openal"); // Force OpenAL sound backendS

            this->doomArgs.push_back("+viz_samp_freq");
            this->doomArgs.push_back(std::to_string(this->audioSamplingFreq));

            this->doomArgs.push_back("+viz_audio_tics");
            this->doomArgs.push_back(std::to_string(this->audioBufferSizeInTics));
        } else if (this->noSound) {
            this->doomArgs.push_back("-nosound");
            this->doomArgs.push_back("+viz_nosound");
            this->doomArgs.push_back("1");
        }

        // notifications buffer
        if (this->notifications) {
            this->doomArgs.push_back("+viz_notifications");
            this->doomArgs.push_back("1");
            this->doomArgs.push_back("+viz_notifications_tics");
            this->doomArgs.push_back(b::lexical_cast<std::string>(this->notificationsBufferSizeInTics));
        }

        // ticrate
        if (this->ticrate != DEFAULT_TICRATE) {
            this->doomArgs.push_back("-ticrate");
            this->doomArgs.push_back(b::lexical_cast<std::string>(this->ticrate));
        }

        // custom args
        for (int i = 0; i < this->customArgs.size(); ++i) {
            this->doomArgs.push_back(customArgs[i]);
        }
    }

    void DoomController::launchDoom() {
        try {
            bpr::child doomProcess = bpr::execute(bpri::set_args(this->doomArgs), bpri::inherit_env());
            #ifdef OS_POSIX
                this->doomProcessPid = doomProcess.pid;
            #endif
            bpr::wait_for_exit(doomProcess);
        }
        catch (...) {
            this->MQController->send(MSG_CODE_DOOM_ERROR, "Unexpected ViZDoom instance crash.");
        }
        this->MQController->send(MSG_CODE_DOOM_PROCESS_EXIT);
    }
}
