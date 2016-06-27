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
#include "boost/process.hpp"

#include <boost/algorithm/string.hpp>
#include <boost/bind.hpp>
#include <boost/chrono.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <climits>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <ctime>

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
        this->MQController = NULL;
        this->MQDoom = NULL;

        /* Shared memory */
        this->InputSMRegion = NULL;
        this->GameStateSMRegion = NULL;
        this->ScreenSMRegion = NULL;

        /* Threads */
        this->signalThread = NULL;
        this->doomThread = NULL;

        /* Flow control */
        this->doomRunning = false;
        this->doomWorking = false;
        this->doomRecordingMap = false;

        this->mapStartTime = 1;
        this->mapTimeout = 0;
        this->mapRestartCount = 0;
        this->mapRestarting = false;
        this->mapLastTic = 1;

        /* Settings */
        this->ticrate = DefaultTicrate;
        #ifdef OS_WIN
            this->exePath = "vizdoom.exe";
        #else
            this->exePath = "vizdoom";
        #endif

        this->iwadPath = "doom2.wad";
        this->filePath = "";
        this->map = "map01";
        this->demoPath = "";
        this->configPath = "";
        this->skill = 3;

        this->screenWidth = 320;
        this->screenHeight = 240;
        this->screenChannels = 3;
        this->screenPitch = 320;
        this->screenSize = 0;
        this->screenDepth = 8;
        this->screenFormat = CRCGCB;
        this->depthBuffer = false;
        this->levelMap = false;
        //this->levelMapMode = NORMAL;
        this->labels = false;

        this->hud = true;
        this->weapon = true;
        this->crosshair = false;
        this->decals = true;
        this->particles = true;

        this->windowHidden = false;
        this->noXServer = false;
        this->noConsole = true;
        this->noSound = true;

        this->allowDoomInput = false;
        this->runDoomAsync = false;

        this->seedDoomRng = true;
        this->doomRngSeed = 0;

        this->instanceRng.seed((unsigned int)bc::high_resolution_clock::now().time_since_epoch().count());

        br::uniform_int_distribution<> rngSeedDist(0, UINT_MAX);
        this->setDoomRngSeed(rngSeedDist(this->instanceRng));

        this->_input = new InputState();
    }

    DoomController::~DoomController() {
        this->close();
        delete _input;
    }


    /* Flow Control */
    /*----------------------------------------------------------------------------------------------------------------*/

    bool DoomController::init() {

        if (!this->doomRunning) {

            try{
                this->generateInstanceId();
                this->MQInit();

                this->signalThread = new b::thread(b::bind(&DoomController::handleSignals, this));

                this->createDoomArgs();
                this->doomThread = new b::thread(b::bind(&DoomController::launchDoom, this));
                this->waitForDoomStart();

                this->SMInit();

                if(this->gameState->VERSION != VIZDOOM_LIB_VERSION){
                    throw ViZDoomMismatchedVersionException(std::string(this->gameState->VERSION_STR), VIZDOOM_LIB_VERSION_STR);
                }

                this->waitForDoomMapStartTime();

                this->MQDoomSend(MSG_CODE_UPDATE);
                this->waitForDoomWork();

                *this->input = *this->_input;

                this->mapLastTic = this->gameState->MAP_TIC;
            }
            catch(...){
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
            this->doomRecordingMap = false;

            this->MQDoomSend(MSG_CODE_CLOSE);
        }

        if (this->signalThread && this->signalThread->joinable()) {
            this->ioService->stop();

            this->signalThread->interrupt();
            this->signalThread->join();
            delete this->signalThread;
            this->signalThread = NULL;

            delete this->ioService;
            this->ioService = NULL;
        }

        if (this->doomThread && this->doomThread->joinable()) {
            this->doomThread->interrupt();
            this->doomThread->join();
            delete this->doomThread;
            this->doomThread = NULL;
        }

        this->SMClose();
        this->MQClose();
    }

    void DoomController::restart() {
        this->close();
        this->init();
    }

    bool DoomController::isTicPossible(){
        if (!this->gameState->GAME_MULTIPLAYER && this->gameState->PLAYER_DEAD) return false;
        else if (this->mapTimeout > 0 && this->gameState->MAP_TIC >= this->mapTimeout + this->mapStartTime) return false;
        else if (this->gameState->MAP_END) return false;
        else return true;
    }

    void DoomController::tic(bool update) {

        if (this->doomRunning) {

            if (this->isTicPossible()) {
                this->mapLastTic = this->gameState->MAP_TIC + 1;
                if(update) this->MQDoomSend(MSG_CODE_TIC_AND_UPDATE);
                else this->MQDoomSend(MSG_CODE_TIC);
                this->waitForDoomWork();
            }
        }
        else throw ViZDoomIsNotRunningException();
    }

    void DoomController::tics(unsigned int tics, bool update){

        if(this->allowDoomInput && !this->runDoomAsync){
            for(int i = 0; i < DeltaButtonCount; ++i){
                this->input->BT_MAX_VALUE[i] = tics * this->_input->BT_MAX_VALUE[i];
            }
        }

        int ticsMade = 0;

        for(int i = 0; i < tics; ++i){
            if(i == tics - 1) this->tic(update);
            else this->tic(false);

            ++ticsMade;

            if(!this->isTicPossible() && i != tics - 1){
                this->MQDoomSend(MSG_CODE_UPDATE);
                this->waitForDoomWork();
                break;
            }
        }

        if(this->allowDoomInput && !this->runDoomAsync){
            for(int i = BinaryButtonCount; i < ButtonCount; ++i){
                this->input->BT_MAX_VALUE[i - BinaryButtonCount] = this->_input->BT_MAX_VALUE[i - BinaryButtonCount];
                this->input->BT[i] = this->input->BT[i]/ticsMade;
            }
        }
    }

    void DoomController::restartMap() {
        this->setMap(this->map);
    }

    void DoomController::respawnPlayer(){

        if(this->doomRunning && !this->mapRestarting && !this->gameState->MAP_END && this->gameState->PLAYER_DEAD){
            if(this->gameState->GAME_MULTIPLAYER){

                bool useAvailable = this->input->BT_AVAILABLE[USE];
                this->input->BT_AVAILABLE[USE] = true;

                do {
                    this->sendCommand(std::string("+use"));

                    this->MQDoomSend(MSG_CODE_TIC);
                    this->waitForDoomWork();

                } while (!this->gameState->MAP_END && this->gameState->PLAYER_DEAD );

                this->sendCommand(std::string("-use"));
                this->MQDoomSend(MSG_CODE_UPDATE);
                this->waitForDoomWork();

                this->input->BT_AVAILABLE[USE] = useAvailable;
                this->mapLastTic = this->gameState->MAP_TIC;

            }
            else this->restartMap();
        }
    }

    void DoomController::sendCommand(std::string command) {
        if(this->doomRunning && this->MQDoom && command.length() <= MQ_MAX_CMD_LEN) this->MQDoomSend(MSG_CODE_COMMAND, command.c_str());
    }

    void DoomController::addCustomArg(std::string arg){
        this->customArgs.push_back(arg);
    }

    void DoomController::clearCustomArgs(){
        this->customArgs.clear();
    }

    bool DoomController::isDoomRunning() { return this->doomRunning; }

    std::string DoomController::getMap(){ return this->map; }

    void DoomController::setMap(std::string map, std::string demoPath) {
        this->map = map;
        this->demoPath = demoPath;

        if (this->doomRunning && !this->mapRestarting) {

            br::uniform_int_distribution<> mapSeedDist(0, UINT_MAX);
            this->setDoomRngSeed(mapSeedDist(this->instanceRng));

            if(this->doomRecordingMap){
                this->sendCommand("stop");
                this->doomRecordingMap = false;
            }

            if(this->gameState->GAME_MULTIPLAYER){
                if(this->gameState->GAME_SETTINGS_CONTROLLER) this->sendCommand(std::string("changemap ") + this->map);
            }
            else if(this->demoPath.length()){
                this->sendCommand(std::string("recordmap ") + this->demoPath + " " + this->map);
                this->doomRecordingMap = true;
            }
            else {
                this->sendCommand(std::string("map ") + this->map);
            }

            if (map != this->map) this->mapRestartCount = 0;
            else ++this->mapRestartCount;

            this->mapRestarting = true;

            this->resetButtons();
            int restartTics = 0;

            bool useAvailable;
            if(this->gameState->GAME_MULTIPLAYER){
                useAvailable = this->input->BT_AVAILABLE[USE];
                this->input->BT_AVAILABLE[USE] = true;

                this->sendCommand(std::string("-use"));
            }

            do {
                ++restartTics;

                if(this->gameState->GAME_MULTIPLAYER){
                    if (restartTics % 2) this->sendCommand(std::string("+use"));
                    else this->sendCommand(std::string("-use"));
                }

                this->MQDoomSend(MSG_CODE_TIC);
                this->waitForDoomWork();

                if(restartTics > 3 && !this->gameState->GAME_MULTIPLAYER){
                    if (this->demoPath.length()) this->sendCommand(std::string("recordmap ") + this->demoPath + " " + this->map);
                    else this->sendCommand(std::string("map ") + this->map);
                    restartTics = 0;
                }

            } while (this->gameState->MAP_END
                     || this->gameState->PLAYER_DEAD
                     || this->gameState->MAP_TIC > this->mapLastTic);

            if(this->gameState->GAME_MULTIPLAYER){
                this->sendCommand(std::string("-use"));
                this->input->BT_AVAILABLE[USE] = useAvailable;
            }

            this->waitForDoomMapStartTime();
            this->MQDoomSend(MSG_CODE_UPDATE);
            this->waitForDoomWork();

            this->mapLastTic = this->gameState->MAP_TIC;
            this->mapRestarting = false;
        }
    }

    void DoomController::playDemo(std::string demoPath){
        if (this->doomRunning && !this->mapRestarting) {

            if(this->doomRecordingMap){
                this->sendCommand("stop");
                this->doomRecordingMap = false;
            }

            this->sendCommand(std::string("playdemo ") + demoPath);

            this->mapRestarting = true;

            this->resetButtons();
            int restartTics = 0;

            do {
                ++restartTics;

                this->MQDoomSend(MSG_CODE_TIC);
                this->waitForDoomWork();

                if(restartTics > 3){
                    this->sendCommand(std::string("playdemo ") + demoPath);
                    restartTics = 0;
                }

            } while (this->gameState->MAP_END
                     || this->gameState->PLAYER_DEAD
                     || this->gameState->MAP_TIC > this->mapLastTic);

            this->waitForDoomMapStartTime();
            this->MQDoomSend(MSG_CODE_UPDATE);
            this->waitForDoomWork();

            this->mapLastTic = this->gameState->MAP_TIC;
            this->mapRestarting = false;
        }
    }


    /* Settings */
    /*----------------------------------------------------------------------------------------------------------------*/

    unsigned int DoomController::getTicrate(){ return this->ticrate; }
    void DoomController::setTicrate(unsigned int ticrate){ this->ticrate = ticrate; }

    std::string DoomController::getExePath() { return this->exePath; }
    void DoomController::setExePath(std::string exePath) { if(!this->doomRunning) this->exePath = exePath; }

    std::string DoomController::getIwadPath() { return this->iwadPath; }
    void DoomController::setIwadPath(std::string iwadPath) { if(!this->doomRunning) this->iwadPath = iwadPath; }

    std::string DoomController::getFilePath() { return this->filePath; }
    void DoomController::setFilePath(std::string filePath) { if(!this->doomRunning) this->filePath = filePath; }

    std::string DoomController::getConfigPath(){ return this->configPath; }
    void DoomController::setConfigPath(std::string configPath) { if(!this->doomRunning) this->configPath = configPath; }

    int DoomController::getSkill(){ return this->skill; }
    void DoomController::setSkill(int skill) {
        if(skill > 4) skill = 4;
        else if(skill < 0) skill = 0;
        this->skill = skill;
        if (this->doomRunning) {
            this->sendCommand(std::string("skill set ") + b::lexical_cast<std::string>(this->skill));
            //this->resetMap();
        }
    }

    unsigned int DoomController::getDoomRngSeed(){
        if (this->doomRunning) return this->gameState->GAME_STATIC_SEED;
        else return this->doomRngSeed;
    }

    void DoomController::setDoomRngSeed(unsigned int seed){
        this->seedDoomRng = true;
        this->doomRngSeed = seed;
        if (this->doomRunning) {
            this->sendCommand(std::string("rngseed set ") + b::lexical_cast<std::string>(this->doomRngSeed));
        }
    }

    void DoomController::clearDoomRngSeed(){
        this->seedDoomRng = false;
        this->doomRngSeed = 0;
        if (this->doomRunning) {
            this->sendCommand("rngseed clear");
        }
    }

    void DoomController::setInstanceRngSeed(unsigned int seed){
        this->instanceRngSeed = seed;
        this->instanceRng.seed(seed);
    }

    unsigned int DoomController::getInstanceRngSeed(){ return this->instanceRngSeed; }

    unsigned int DoomController::getMapStartTime() { return this->mapStartTime; }
    void DoomController::setMapStartTime(unsigned int tics) { this->mapStartTime = tics ? tics : 1; }

    unsigned int DoomController::getMapTimeout() { return this->mapTimeout; }
    void DoomController::setMapTimeout(unsigned int tics) { this->mapTimeout = tics; }

    bool DoomController::isMapFirstTic() {
        if (this->doomRunning && this->gameState->MAP_TIC <= 1) return true;
        else return false;
    }

    bool DoomController::isMapLastTic() {
        if (this->doomRunning && this->mapTimeout > 0 && this->gameState->MAP_TIC >= this->mapTimeout + this->mapStartTime) return true;
        else return false;
    }

    bool DoomController::isMapEnded() {
        if (this->doomRunning && this->gameState->MAP_END) return true;
        else return false;
    }

    unsigned int DoomController::getMapLastTic() {
        return this->mapLastTic;
    }

    void DoomController::setNoConsole(bool console) {
        if(!this->doomRunning) this->noConsole = console;
    }

    void DoomController::setNoSound(bool sound) {
        if(!this->doomRunning) this->noSound = sound;
    }

    void DoomController::setScreenResolution(unsigned int width, unsigned int height) {
        if(!this->doomRunning) {
            this->screenWidth = width;
            this->screenHeight = height;
        }
    }

    void DoomController::setScreenWidth(unsigned int width) { if(!this->doomRunning) this->screenWidth = width; }
    void DoomController::setScreenHeight(unsigned int height) { if(!this->doomRunning) this->screenHeight = height; }
    void DoomController::setScreenFormat(ScreenFormat format) {
        if(!this->doomRunning) {
            this->screenFormat = format;
            switch (format) {
                case CRCGCB:
                case RGB24:
                case CBCGCR:
                case BGR24:
                    this->screenChannels = 3;
                    break;
                case CRCGCBDB:
                case RGBA32:
                case ARGB32:
                case CBCGCRDB:
                case BGRA32:
                case ABGR32:
                    this->screenChannels = 4;
                    break;
                case GRAY8:
                case DEPTH_BUFFER8:
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
                case CRCGCBDB:
                case CBCGCRDB:
                case GRAY8:
                case DEPTH_BUFFER8:
                case DOOM_256_COLORS8:
                    this->screenDepth = 8;
                    break;
                default:
                    this->screenDepth = 0;
            }
        }
    }

    void DoomController::setWindowHidden(bool windowHidden){ if(!this->doomRunning) this->windowHidden = windowHidden; }
    void DoomController::setNoXServer(bool noXServer) { if(!this->doomRunning) this->noXServer = noXServer; }

    void DoomController::setRenderHud(bool hud) {
        this->hud = hud;
        if (this->doomRunning) {
            if (this->hud) this->sendCommand("screenblocks 10");
            else this->sendCommand("screenblocks 12");
        }
    }

    void DoomController::setRenderWeapon(bool weapon) {
        this->weapon = weapon;
        if (this->doomRunning) {
            if (this->weapon) this->sendCommand("r_drawplayersprites 1");
            else this->sendCommand("r_drawplayersprites 1");
        }
    }

    void DoomController::setRenderCrosshair(bool crosshair) {
        this->crosshair = crosshair;
        if (this->doomRunning) {
            if (this->crosshair) {
                this->sendCommand("crosshairhealth false");
                this->sendCommand("crosshair 1");
            }
            else this->sendCommand("crosshair 0");
        }
    }

    void DoomController::setRenderDecals(bool decals) {
        this->decals = decals;
        if (this->doomRunning) {
            if (this->decals) this->sendCommand("cl_maxdecals 1024");
            else this->sendCommand("cl_maxdecals 0");
        }
    }

    void DoomController::setRenderParticles(bool particles) {
        this->particles = particles;
        if (this->doomRunning) {
            if (this->particles) this->sendCommand("r_particles 1");
            else this->sendCommand("r_particles 0");
        }
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
        if (this->doomRunning) return (size_t) this->gameState->SCREEN_PITCH;
        else return (size_t) this->screenDepth/8*this->screenWidth;
    }

    ScreenFormat DoomController::getScreenFormat() {
        if (this->doomRunning) return (ScreenFormat) this->gameState->SCREEN_FORMAT;
        else return this->screenFormat;
    }

    size_t DoomController::getScreenSize() {
        if (this->doomRunning) return (size_t) this->gameState->SCREEN_SIZE;
        else return (size_t) this->screenChannels * this->screenWidth * this->screenHeight;
    }

    /* SM setters & getters */
    /*----------------------------------------------------------------------------------------------------------------*/

    uint8_t *const DoomController::getScreen() { return this->screen; }

    DoomController::InputState *const DoomController::getInput() { return this->input; }

    DoomController::GameState *const DoomController::getGameState() { return this->gameState; }

    int DoomController::getButtonState(Button button){
        if(this->doomRunning) return this->input->BT[button];
        else return 0;
    }

    void DoomController::setButtonState(Button button, int state) {
        if (button < ButtonCount && button >= 0 && this->doomRunning)
            this->input->BT[button] = state;

    }

    void DoomController::toggleButtonState(Button button) {
        if (button < ButtonCount && button >= 0 && this->doomRunning)
            this->input->BT[button] = !this->input->BT[button];

    }

    bool DoomController::isButtonAvailable(Button button){
        if(this->doomRunning) return this->input->BT_AVAILABLE[button];
        else return this->_input->BT_AVAILABLE[button];
    }

    void DoomController::setButtonAvailable(Button button, bool allow) {
        if (button < ButtonCount && button >= 0) {
            if (this->doomRunning) this->input->BT_AVAILABLE[button] = allow;
            this->_input->BT_AVAILABLE[button] = allow;
        }
    }

    void DoomController::resetButtons(){
        if (this->doomRunning)
            for (int i = 0; i < ButtonCount; ++i)
                this->input->BT[i] = 0;
    }

    void DoomController::disableAllButtons(){
        for (int i = 0; i < ButtonCount; ++i){
            if (this->doomRunning) this->input->BT_AVAILABLE[i] = false;
            this->_input->BT_AVAILABLE[i] = false;
        }
    }

    void DoomController::availableAllButtons(){
        for (int i = 0; i < ButtonCount; ++i){
            if (this->doomRunning) this->input->BT_AVAILABLE[i] = true;
            this->_input->BT_AVAILABLE[i] = true;
        }
    }

    void DoomController::setButtonMaxValue(Button button, unsigned int value){
        if(button >= BinaryButtonCount){
            if (this->doomRunning) this->input->BT_MAX_VALUE[button - BinaryButtonCount] = value;
            this->_input->BT_MAX_VALUE[button - BinaryButtonCount] = value;
        }
    }

    int DoomController::getButtonMaxValue(Button button){
        if(button >= BinaryButtonCount){
            if (this->doomRunning) return this->input->BT_MAX_VALUE[button - BinaryButtonCount];
            else return this->_input->BT_MAX_VALUE[button - BinaryButtonCount];
        }
        else return 1;
    }

    bool DoomController::isAllowDoomInput(){ return this->allowDoomInput; }
    void DoomController::setAllowDoomInput(bool set){ if(!this->doomRunning) this->allowDoomInput = set; }

    bool DoomController::isRunDoomAsync(){ return this->runDoomAsync; }
    void DoomController::setRunDoomAsync(bool set){ if(!this->doomRunning) this->runDoomAsync = set; }

    /* GameVariables getters */
    /*----------------------------------------------------------------------------------------------------------------*/

    int DoomController::getGameVariable(GameVariable var) {
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
                return this->gameState->PLAYER_ON_GROUND;
            case ATTACK_READY :
                return this->gameState->PLAYER_ATTACK_READY;
            case ALTATTACK_READY :
                return this->gameState->PLAYER_ALTATTACK_READY;
            case SELECTED_WEAPON :
                return this->gameState->PLAYER_SELECTED_WEAPON;
            case SELECTED_WEAPON_AMMO :
                return this->gameState->PLAYER_SELECTED_WEAPON_AMMO;
            case PLAYER_NUMBER:
                return this->gameState->PLAYER_NUMBER;
            case PLAYER_COUNT:
                return this->gameState->PLAYER_COUNT;
        }
        if(var >= AMMO0 && var <= AMMO9){
            return this->gameState->PLAYER_AMMO[var - AMMO0];
        }
        else if(var >= WEAPON0 && var <= WEAPON9){
            return this->gameState->PLAYER_WEAPON[var - WEAPON0];
        }
        else if(var >= USER1 && var <= USER30){
            return this->gameState->MAP_USER_VARS[var - USER1];
        }
        else if(var >= PLAYER1_FRAGCOUNT && var <= PLAYER8_FRAGCOUNT){
            return this->gameState->PLAYERS_FRAGCOUNT[var - PLAYER1_FRAGCOUNT];
        }
        else return 0;
    }

    int DoomController::getGameTic() { return this->gameState->GAME_TIC; }
    bool DoomController::isMultiplayerGame() { return this->gameState->GAME_MULTIPLAYER; }
    bool DoomController::isNetGame() { return this->gameState->GAME_NETGAME; }
    int DoomController::getMapTic() { return this->gameState->MAP_TIC; }

    int DoomController::getMapReward() { return this->gameState->MAP_REWARD; }

    int DoomController::getMapKillCount() { return this->gameState->MAP_KILLCOUNT; }
    int DoomController::getMapItemCount() { return this->gameState->MAP_ITEMCOUNT; }
    int DoomController::getMapSecretCount() { return this->gameState->MAP_SECRETCOUNT; }

    bool DoomController::isPlayerDead() { return this->gameState->PLAYER_DEAD; }

    int DoomController::getPlayerKillCount() { return this->gameState->PLAYER_KILLCOUNT; }
    int DoomController::getPlayerItemCount() { return this->gameState->PLAYER_ITEMCOUNT; }
    int DoomController::getPlayerSecretCount() { return this->gameState->PLAYER_SECRETCOUNT; }
    int DoomController::getPlayerFragCount() { return this->gameState->PLAYER_FRAGCOUNT; }
    int DoomController::getPlayerDeathCount() { return this->gameState->PLAYER_DEATHCOUNT; }

    int DoomController::getPlayerHealth() { return this->gameState->PLAYER_HEALTH; }
    int DoomController::getPlayerArmor() { return this->gameState->PLAYER_ARMOR; }

    bool DoomController::isPlayerOnGround() { return this->gameState->PLAYER_ON_GROUND; }
    bool DoomController::isPlayerAttackReady() { return this->gameState->PLAYER_ATTACK_READY; }
    bool DoomController::isPlayerAltAttackReady() { return this->gameState->PLAYER_ALTATTACK_READY; }

    int DoomController::getPlayerSelectedWeaponAmmo() { return this->gameState->PLAYER_SELECTED_WEAPON_AMMO; }
    int DoomController::getPlayerSelectedWeapon() { return this->gameState->PLAYER_SELECTED_WEAPON; }

    int DoomController::getPlayerAmmo(unsigned int slot) {
        return slot < SlotCount  ? this->gameState->PLAYER_AMMO[slot] : 0;
    }

    int DoomController::getPlayerWeapon(unsigned int slot) {
        return slot < SlotCount  ? this->gameState->PLAYER_WEAPON[slot] : 0;
    }


    /* Protected and private functions */
    /*----------------------------------------------------------------------------------------------------------------*/

    void DoomController::generateInstanceId(){
        std::string chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";
        this->instanceId = "";

        br::uniform_int_distribution<> charDist(0, chars.length() - 1);
        br::mt19937 rng;
        rng.seed((unsigned int)bc::high_resolution_clock::now().time_since_epoch().count());

        for (int i = 0; i < INSTANCE_ID_LENGHT; ++i) {
            this->instanceId += chars[charDist(rng)];
        }
    }


    /* Signals */
    /*----------------------------------------------------------------------------------------------------------------*/

    void DoomController::handleSignals(){
        this->ioService = new ba::io_service();
        ba::signal_set signals(*this->ioService, SIGINT, SIGABRT, SIGTERM);
        signals.async_wait(b::bind(signalHandler, b::ref(signals), this, _1, _2));

        this->ioService->run();
    }

    void DoomController::signalHandler(ba::signal_set& signal, DoomController* controller, const bs::error_code& error, int sigNumber){
        controller->intSignal(sigNumber);
    }

    void DoomController::intSignal(int sigNumber){
        this->MQDoomSend(MSG_CODE_CLOSE);
        this->MQControllerSend(MSG_CODE_SIG + sigNumber);
    }

    /* Flow */
    /*----------------------------------------------------------------------------------------------------------------*/

    void DoomController::waitForDoomStart() {

        this->doomWorking = true;

        Message msg;

        unsigned int priority;
        bip::message_queue::size_type recv_size;

        this->MQControllerRecv(&msg, recv_size, priority);
        switch (msg.code) {
            case MSG_CODE_DOOM_DONE :
                this->doomRunning = true;
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
                throw ViZDoomSignalException("SIGINT");

            case MSG_CODE_SIGABRT :
                this->close();
                throw ViZDoomSignalException("SIGABRT");

            case MSG_CODE_SIGTERM :
                this->close();
                throw ViZDoomSignalException("SIGTERM");
        }

        this->doomWorking = false;
    }

    void DoomController::waitForDoomWork() {

        if(doomRunning){
            this->doomWorking = true;

            Message msg;

            unsigned int priority;
            bip::message_queue::size_type recv_size;

            bool done = false;
            do {
                this->MQControllerRecv(&msg, recv_size, priority);
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
                        throw ViZDoomSignalException("SIGINT");

                    case MSG_CODE_SIGABRT :
                        this->close();
                        throw ViZDoomSignalException("SIGABRT");

                    case MSG_CODE_SIGTERM :
                        this->close();
                        throw ViZDoomSignalException("SIGTERM");
                }
            } while (!done);

            this->doomWorking = false;
        }
        else throw ViZDoomIsNotRunningException();
    }

    void DoomController::waitForDoomMapStartTime() {
        while(this->gameState->MAP_TIC < this->mapStartTime) {
            this->MQDoomSend(MSG_CODE_TIC);
            this->waitForDoomWork();
        }
    }

    /* Init */
    /*----------------------------------------------------------------------------------------------------------------*/

    void DoomController::createDoomArgs(){
        this->doomArgs.clear();

        //exe
        if(!bfs::exists(this->exePath) || bfs::is_directory(this->exePath)){
        #ifdef OS_WIN
            if(!bfs::exists(this->exePath + ".exe")) throw FileDoesNotExistException(this->exePath);
            this->exePath += ".exe";
        #else
            throw FileDoesNotExistException(this->exePath);
        #endif
        }
        this->doomArgs.push_back(this->exePath);

        //main wad
        if(this->iwadPath.length() != 0){
            if(!bfs::exists(this->iwadPath) || bfs::is_directory(this->exePath)) throw FileDoesNotExistException(this->iwadPath);
            this->doomArgs.push_back("-iwad");
            this->doomArgs.push_back(this->iwadPath);
        }

        //wads
        if (this->filePath.length() != 0) {
            if(!bfs::exists(this->filePath) || bfs::is_directory(this->exePath)) throw FileDoesNotExistException(this->filePath);
            
            this->doomArgs.push_back("-file");
            this->doomArgs.push_back(this->filePath);
        }

        this->doomArgs.push_back("-config");
        if (this->configPath.length() != 0) this->doomArgs.push_back(this->configPath);
        else this->doomArgs.push_back("_vizdoom.ini");

        if(this->seedDoomRng) {
            this->doomArgs.push_back("+rngseed");
            this->doomArgs.push_back(b::lexical_cast<std::string>(this->doomRngSeed));
        }

        //map
        this->doomArgs.push_back("+map");
        if(this->map.length() > 0) this->doomArgs.push_back(this->map);
        else this->doomArgs.push_back("map01");

        //if (this->demoPath.length() != 0){
        //    this->doomArgs.push_back("-record");
        //    this->doomArgs.push_back(this->demoPath);
        //    this->doomRecordingMap = true;
        //}
        //else this->doomRecordingMap = false;

        //this->doomArgs.push_back("+viz_loop_map");
        //this->doomArgs.push_back("1");

        //skill
        this->doomArgs.push_back("-skill");
        this->doomArgs.push_back(b::lexical_cast<std::string>(this->skill));

        //resolution and aspect ratio

        this->doomArgs.push_back("-width");
        this->doomArgs.push_back(b::lexical_cast<std::string>(this->screenWidth));
        //this->doomArgs.push_back("+vid_defwidth");
        //this->doomArgs.push_back(b::lexical_cast<std::string>(this->screenWidth));

        this->doomArgs.push_back("-height");
        this->doomArgs.push_back(b::lexical_cast<std::string>(this->screenHeight));
        //this->doomArgs.push_back("+vid_defheight");
        //this->doomArgs.push_back(b::lexical_cast<std::string>(this->screenHeight));

        float ratio = this->screenWidth/this->screenHeight;

        this->doomArgs.push_back("+vid_aspect");
        if(ratio == 16.0/9.0) this->doomArgs.push_back("1");
        else if(ratio == 16.0/10.0) this->doomArgs.push_back("2");
        else if(ratio == 4.0/3.0) this->doomArgs.push_back("3");
        else if(ratio == 5.0/4.0) this->doomArgs.push_back("4");
        else this->doomArgs.push_back("0");

        //hud
        this->doomArgs.push_back("+screenblocks");
        if (this->hud) this->doomArgs.push_back("10");
        else this->doomArgs.push_back("12");

        //weapon
        this->doomArgs.push_back("+r_drawplayersprites");
        if (this->weapon) this->doomArgs.push_back("1");
        else this->doomArgs.push_back("0");

        //crosshair
        this->doomArgs.push_back("+crosshair");
        if (this->crosshair) {
            this->doomArgs.push_back("1");
            this->doomArgs.push_back("+crosshairhealth");
            this->doomArgs.push_back("0");
        }
        else this->doomArgs.push_back("0");

        //decals
        this->doomArgs.push_back("+cl_maxdecals");
        if (this->decals) this->doomArgs.push_back("1024");
        else this->doomArgs.push_back("0");

        //particles
        this->doomArgs.push_back("+r_particles");
        if (this->decals) this->doomArgs.push_back("1");
        else this->doomArgs.push_back("0");

        //window mode
        this->doomArgs.push_back("+fullscreen");
        this->doomArgs.push_back("0");

        //weapon auto switch
        //this->doomArgs.push_back("+neverswitchonpickup");
        //this->doomArgs.push_back("1");

        //vizdoom this->doomArgs
        this->doomArgs.push_back("+viz_controlled");
        this->doomArgs.push_back("1");

        this->doomArgs.push_back("+viz_instance_id");
        this->doomArgs.push_back(this->instanceId);

        if(this->noConsole){
            this->doomArgs.push_back("+viz_noconsole");
            this->doomArgs.push_back("1");
        }

        if(this->allowDoomInput){
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
        }
        else{
            //disable mouse
            this->doomArgs.push_back("+use_mouse");
            this->doomArgs.push_back("0");
        }

        if(this->runDoomAsync){
            this->doomArgs.push_back("+viz_async");
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

        //no wipe animation
        this->doomArgs.push_back("+wipetype");
        this->doomArgs.push_back("0");

        //idle/joy
        this->doomArgs.push_back("-noidle");
        this->doomArgs.push_back("-nojoy");

        //sound
        if(this->noSound){
            this->doomArgs.push_back("-nosound");
            this->doomArgs.push_back("+viz_nosound");
            this->doomArgs.push_back("1");
        }

        if(this->ticrate != DefaultTicrate){
            this->doomArgs.push_back("-ticrate");
            this->doomArgs.push_back(b::lexical_cast<std::string>(this->ticrate));
        }

        //fps = ticrate and no vsync
        this->doomArgs.push_back("+cl_capfps");
        this->doomArgs.push_back("1");

        this->doomArgs.push_back("+vid_vsync");
        this->doomArgs.push_back("0");

        //custom args
        for(int i = 0; i < this->customArgs.size(); ++i){
            this->doomArgs.push_back(customArgs[i]);
        }
    }

    void DoomController::launchDoom() {
        try{
            bpr::child doomProcess = bpr::execute(bpri::set_args(this->doomArgs), bpri::inherit_env());
            bpr::wait_for_exit(doomProcess);
        }
        catch(...){
            this->MQControllerSend(MSG_CODE_DOOM_ERROR, "Unexpected ViZDoom instance crash.");
        }
        this->MQControllerSend(MSG_CODE_DOOM_PROCESS_EXIT);
    }

    /* Shared memory */
    /*----------------------------------------------------------------------------------------------------------------*/

    void DoomController::SMInit() {
        this->SMName = std::string(SM_NAME_BASE) + instanceId;
        //bip::shared_memory_object::remove(this->SMName.c_str());
        try {
            this->SM = bip::shared_memory_object(bip::open_only, this->SMName.c_str(), bip::read_write);
            this->SM.get_size(this->SMSize);

            size_t SMGameStateAddress = 0;
            this->GameStateSMRegion = new bip::mapped_region(this->SM, bip::read_only, SMGameStateAddress, sizeof(DoomController::GameState));
            this->gameState = static_cast<DoomController::GameState *>(this->GameStateSMRegion->get_address());

            size_t SMInputAddress = sizeof(DoomController::GameState);
            this->InputSMRegion = new bip::mapped_region(this->SM, bip::read_write, SMInputAddress, sizeof(DoomController::InputState));
            this->input = static_cast<DoomController::InputState *>(this->InputSMRegion->get_address());
            
            this->screenWidth = this->gameState->SCREEN_WIDTH;
            this->screenHeight = this->gameState->SCREEN_HEIGHT;
            this->screenPitch = this->gameState->SCREEN_PITCH;
            this->screenSize = this->gameState->SCREEN_SIZE;
            this->screenFormat = (ScreenFormat)this->gameState->SCREEN_FORMAT;

            size_t SMScreenAddress = sizeof(DoomController::GameState) + sizeof(DoomController::InputState);
            this->ScreenSMRegion = new bip::mapped_region(this->SM, bip::read_only, SMScreenAddress, this->screenSize);
            this->screen = static_cast<uint8_t *>(this->ScreenSMRegion->get_address());
        }
        catch(...) { //bip::interprocess_exception
            throw SharedMemoryException("Failed to open shared memory.");
        }

        size_t SMExpectedSize = sizeof(DoomController::GameState) + sizeof(DoomController::InputState) + this->screenSize;
        if(this->gameState->SM_SIZE != this->SMSize) throw SharedMemoryException("Memory size does not match the the expected size.");
    }

    void DoomController::SMClose() {
        bip::shared_memory_object::remove(this->SMName.c_str());

        if(this->InputSMRegion) {
            delete this->InputSMRegion;
            this->InputSMRegion = NULL;
        }
        if(this->GameStateSMRegion) {
            delete this->GameStateSMRegion;
            this->GameStateSMRegion = NULL;
        }
        if(this->ScreenSMRegion) {
            delete this->ScreenSMRegion;
            this->ScreenSMRegion = NULL;
        }
    }


    /* Message queues */
    /*----------------------------------------------------------------------------------------------------------------*/

    void DoomController::MQInit() {

        this->MQControllerName = std::string(MQ_NAME_CTR_BASE) + this->instanceId;
        this->MQDoomName = std::string(MQ_NAME_DOOM_BASE) + this->instanceId;

        try {
            bip::message_queue::remove(this->MQControllerName.c_str());
            bip::message_queue::remove(this->MQDoomName.c_str());

            this->MQController = new bip::message_queue(bip::open_or_create, this->MQControllerName.c_str(), MQ_MAX_MSG_NUM, MQ_MAX_MSG_SIZE);
            this->MQDoom = new bip::message_queue(bip::open_or_create, this->MQDoomName.c_str(), MQ_MAX_MSG_NUM, MQ_MAX_MSG_SIZE);
        }
        catch(...) { // bip::interprocess_exception
            throw MessageQueueException("Failed to create message queues.");
        }
    }

    void DoomController::MQControllerSend(uint8_t code, const char *command) {
        Message msg;
        msg.code = code;
        if(command != NULL) strncpy(msg.command, command, MQ_MAX_CMD_LEN);
        try {
            this->MQController->send(&msg, sizeof(Message), 0);
        }
        catch(...){ // bip::interprocess_exception
            throw MessageQueueException("Failed to send message.");
        }
    }

    void DoomController::MQDoomSend(uint8_t code, const char *command) {
        Message msg;
        msg.code = code;
        if(command != NULL) strncpy(msg.command, command, MQ_MAX_CMD_LEN);
        try{
            this->MQDoom->send(&msg, sizeof(Message), 0);
        }
        catch(...){ // bip::interprocess_exception
            throw MessageQueueException("Failed to send message.");
        }
    }

    void DoomController::MQControllerRecv(void *msg, size_t &size, unsigned int &priority) {
        try {
            this->MQController->receive(msg, sizeof(Message), size, priority);
        }
        catch(...){ // bip::interprocess_exception
            throw MessageQueueException("Failed to receive message.");
        }
    }

    void DoomController::MQClose() {
        bip::message_queue::remove(this->MQDoomName.c_str());
        if(this->MQDoom) {
            delete this->MQDoom;
            this->MQDoom = NULL;
        }

        bip::message_queue::remove(this->MQControllerName.c_str());
        if(this->MQController) {
            delete this->MQController;
            this->MQController = NULL;
        }
    }
}
