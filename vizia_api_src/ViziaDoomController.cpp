#include "ViziaDoomController.h"

#include <vector>
#include <iostream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <cstdio>

namespace Vizia {

//PUBLIC FUNCTIONS

    void signalHandler(ba::signal_set& signal, DoomController* controller, const bs::error_code& error, int signal_number){
        controller->intSignal();
    }

    DoomController::DoomController() {

        this->MQController = NULL;
        this->MQDoom = NULL;

        this->InputSMRegion = NULL;
        this->GameVariablesSMRegion = NULL;
        this->ScreenSMRegion = NULL;

        this->screenWidth = 320;
        this->screenHeight = 240;
        this->screenChannels = 3;
        this->screenPitch = 320;
        this->screenSize = 0;
        this->screenDepth = 8;
        this->screenFormat = CRCGCB;

        this->gamePath = "viziazdoom";
        this->iwadPath = "doom2.wad";
        this->filePath = "";
        this->map = "map01";
        this->configPath = "";
        this->skill = 0;

        this->hud = true;
        this->weapon = true;
        this->crosshair = false;
        this->decals = true;
        this->particles = true;

        this->windowHidden = false;
        this->noXServer = false;
        this->noConsole = true;

        // AUTO RESTART && TIMEOUT
        this->autoRestart = false;
        this->autoRestartOnTimeout = true;
        this->autoRestartOnPlayersDeath = true;
        this->autoRestartOnMapEnd = true;
        this->mapStartTime = 1;
        this->mapTimeout = 0;
        this->mapRestartCount = 0;
        this->mapRestarting = false;
        this->mapEnded = false;
        this->mapLastTic = 1;

        this->allowDoomInput = false;
        this->runDoomAsync = false;

        //SEED
        this->generateInstanceId();
        //this->generateStaticSeed();
        this->useStaticSeed = false;
        this->staticSeed = 0;
        this->doomRunning = false;
        this->doomWorking = false;

        //THREADS
        this->signalThread = NULL;
        this->doomThread = NULL;

        this->_input = new InputStruct();
    }

    DoomController::~DoomController() {
        //this->close();
        delete _input;
    }

//FLOW CONTROL

    bool DoomController::init() {

        if (!this->doomRunning && this->iwadPath.length() != 0 && this->map.length() != 0) {

            try{
                this->doomRunning = true;

                if (this->instanceId.length() == 0) generateInstanceId();
                this->MQInit();

                this->signalThread = new b::thread(b::bind(&DoomController::handleSignals, this));

                this->doomThread = new b::thread(b::bind(&DoomController::launchDoom, this));
                this->waitForDoomStart();

                this->SMInit();
                this->waitForDoomMapStartTime();

                this->MQDoomSend(MSG_CODE_UPDATE);
                this->waitForDoomWork();

                *this->input = *this->_input;

                this->lastTicTime = std::clock();
                this->mapLastTic = this->gameVariables->MAP_TIC;
            }
            catch(const Exception &e){
                this->doomRunning = false;
                this->close();
                throw;
            }
        }

        return this->doomRunning;
    }

    void DoomController::close() {

        if (this->doomRunning) {

            this->doomRunning = false;
            if (this->signalThread && this->signalThread->joinable()) {
                this->ioService.stop();
                this->signalThread->interrupt();
                this->signalThread->join();
            }

            this->MQDoomSend(MSG_CODE_CLOSE);

            if (this->doomThread && this->doomThread->joinable()) {
                this->doomThread->interrupt();
                this->doomThread->join();
            }

            this->SMClose();
            this->MQClose();
        }
    }

    void DoomController::intSignal(){
        this->MQControllerSend(MSG_CODE_SIGNAL_INT_ABRT_TERM);
    }

    void DoomController::restart() {
        this->close();
        this->init();
    }

    bool DoomController::tic() {
        this->tic(true);
    }

    bool DoomController::tic(bool update) {

        if (this->doomRunning) {

            if (!this->mapEnded) {

                //this->lastTicTime = bc::steady_clock::now();
                this->lastTicTime = std::clock();

                this->mapLastTic = this->gameVariables->MAP_TIC + 1;
                if(update) this->MQDoomSend(MSG_CODE_TIC_N_UPDATE);
                else this->MQDoomSend(MSG_CODE_TIC);
                this->waitForDoomWork();
            }

            if (this->gameVariables->PLAYER_DEAD) {
                this->mapEnded = true;
                if (this->autoRestart && this->autoRestartOnPlayersDeath) this->restartMap();
            }
            else if (this->mapTimeout > 0 && this->gameVariables->MAP_TIC >= this->mapTimeout + this->mapStartTime) {
                this->mapEnded = true;
                if (this->autoRestart && this->autoRestartOnTimeout) this->restartMap();
            }
            else if (this->gameVariables->MAP_END) {
                this->mapEnded = true;
                if (this->autoRestart && this->autoRestartOnMapEnd) this->restartMap();
            }
        }
        else throw DoomIsNotRunningException();

        return !this->mapEnded;
    }

    bool DoomController::tics(unsigned int tics){
        this->tics(tics, true);
    }

    bool DoomController::tics(unsigned int tics, bool update){
        bool lastTic = this->mapEnded;

        if(this->allowDoomInput && !this->runDoomAsync){
            for(int i = 0; i < DeltaButtonsNumber; ++i){
                this->input->BT_MAX_VALUE[i] = tics * this->_input->BT_MAX_VALUE[i];
            }
        }

        int ticsMade = 0;

        for(int i = 0; i < tics; ++i){
            if(i == tics - 1) lastTic = this->tic(update);
            else lastTic = this->tic(false);

            ++ticsMade;

            if(!lastTic){
                this->MQDoomSend(MSG_CODE_UPDATE);
                this->waitForDoomWork();
                break;
            }

            //if(i == 0) this->resetDescreteButtons();
        }

        if(this->allowDoomInput && !this->runDoomAsync){
            for(int i = DiscreteButtonsNumber; i < ButtonsNumber; ++i){
                this->input->BT_MAX_VALUE[i - DiscreteButtonsNumber] = this->_input->BT_MAX_VALUE[i - DiscreteButtonsNumber];
                this->input->BT[i] = this->input->BT[i]/ticsMade;
            }
        }

        return lastTic;
    }

    void DoomController::restartMap() {
        this->setMap(this->map);
    }

    void DoomController::resetMap() {
        this->restartMap();
    }

    void DoomController::sendCommand(std::string command) {
        if(command.length() <= MQ_MAX_CMD_LEN) this->MQDoomSend(MSG_CODE_COMMAND, command.c_str());
    }

    void DoomController::addCustomArg(std::string arg){
        this->customArgs.push_back(arg);
    }

    void DoomController::clearCustomArgs(){
        this->customArgs.clear();
    }

    bool DoomController::isDoomRunning() { return this->doomRunning; }

//SETTINGS

//GAME & MAP SETTINGS

    std::string DoomController::getInstanceId() { return this->instanceId; }
    void DoomController::setInstanceId(std::string id) { if(!this->doomRunning) this->instanceId = id; }

    std::string DoomController::getGamePath() { return this->gamePath; }
    void DoomController::setGamePath(std::string path) { if(!this->doomRunning) this->gamePath = path; }

    std::string DoomController::getIwadPath() { return this->iwadPath; }
    void DoomController::setIwadPath(std::string path) { if(!this->doomRunning) this->iwadPath = path; }

    std::string DoomController::getFilePath() { return this->filePath; }
    void DoomController::setFilePath(std::string path) { if(!this->doomRunning) this->filePath = path; }

    std::string DoomController::getConfigPath(){ return this->configPath; }
    void DoomController::setConfigPath(std::string path) { if(!this->doomRunning) this->configPath = path; }

    std::string DoomController::getMap(){ return this->map; }
    void DoomController::setMap(std::string map) {
        this->map = map;
        if (this->doomRunning && !this->mapRestarting) {
            this->sendCommand(std::string("map ") + this->map);
            if (map != this->map) this->mapRestartCount = 0;
            else ++this->mapRestartCount;

            this->mapRestarting = true;

            this->resetButtons();

            int restartingTics = 0;

            do {
                ++restartingTics;
                this->MQDoomSend(MSG_CODE_TIC);
                this->waitForDoomWork();

                if (restartingTics > 4) {
                    this->sendCommand(std::string("map ") + this->map);
                    restartingTics = 0;
                }

            } while (this->gameVariables->MAP_END || this->gameVariables->MAP_TIC > 1);

            this->waitForDoomMapStartTime();

            this->MQDoomSend(MSG_CODE_UPDATE);
            this->waitForDoomWork();

            this->mapLastTic = this->gameVariables->MAP_TIC;

            this->mapRestarting = false;
            this->mapEnded = false;
        }
    }

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

    unsigned int DoomController::getSeed(){
        if (this->doomRunning) return this->gameVariables->GAME_SEED;
        else return 0;
    }

    unsigned int DoomController::getStaticSeed(){
        if (this->doomRunning) return this->gameVariables->GAME_STATIC_SEED;
        else return this->staticSeed;
    }

    void DoomController::setStaticSeed(unsigned int seed){
        if(seed == 0){
            this->useStaticSeed = false;
            this->staticSeed = 0;
            if (this->doomRunning) {
                this->sendCommand("rngseed clear");
            }
        }
        else {
            this->useStaticSeed = true;
            this->staticSeed = seed;
            if (this->doomRunning) {
                this->sendCommand(std::string("rngseed set ") + b::lexical_cast<std::string>(this->staticSeed));
            }
        }
    }

    void DoomController::setUseStaticSeed(bool set){ this->useStaticSeed = true; }
    bool DoomController::isUseStaticSeed(){ return this->useStaticSeed; }

    void DoomController::setAutoMapRestart(bool set) { this->autoRestart = set; }
    void DoomController::setAutoMapRestartOnTimeout(bool set) { this->autoRestartOnTimeout = set; }
    void DoomController::setAutoMapRestartOnPlayerDeath(bool set) { this->autoRestartOnPlayersDeath = set; }
    void DoomController::setAutoMapRestartOnMapEnd(bool set) { this->autoRestartOnMapEnd = set; }

    unsigned int DoomController::getMapStartTime() { return this->mapStartTime; }
    void DoomController::setMapStartTime(unsigned int tics) { this->mapStartTime = tics ? tics : 1; }

    unsigned int DoomController::getMapTimeout() { return this->mapTimeout; }
    void DoomController::setMapTimeout(unsigned int tics) { this->mapTimeout = tics; }

    bool DoomController::isMapFirstTic() {
        if (this->doomRunning && this->gameVariables->MAP_TIC <= 1) return true;
        else return false;
    }

    bool DoomController::isMapLastTic() {
        if (this->doomRunning && this->mapTimeout > 0 && this->gameVariables->MAP_TIC >= this->mapTimeout + this->mapStartTime) return true;
        else return false;
    }

    bool DoomController::isMapEnded() {
        if (this->doomRunning && this->gameVariables->MAP_END) return true;
        else return false;
    }

    unsigned int DoomController::getMapLastTic() {
        return this->mapLastTic;
    }

    void DoomController::setNoConsole(bool console) {
        if(!this->doomRunning) this->noConsole=console;
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
                case CRCGCBZB:
                case RGBA32:
                case ARGB32:
                case CBCGCRZB:
                case BGRA32:
                case ABGR32:
                    this->screenChannels = 4;
                    break;
                case GRAY8:
                case ZBUFFER8:
                case DOOM_256_COLORS:
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
                case CRCGCBZB:
                case CBCGCRZB:
                case GRAY8:
                case ZBUFFER8:
                case DOOM_256_COLORS:
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
        if (this->doomRunning) return this->gameVariables->SCREEN_WIDTH;
        else return this->screenWidth;
    }

    unsigned int DoomController::getScreenHeight() {
        if (this->doomRunning) return this->gameVariables->SCREEN_HEIGHT;
        else return this->screenHeight;
    }

    unsigned int DoomController::getScreenChannels() { return this->screenChannels; }

    unsigned int DoomController::getScreenDepth() { return this->screenDepth; }

    size_t DoomController::getScreenPitch() {
        if (this->doomRunning) return (size_t) this->gameVariables->SCREEN_PITCH;
        else return (size_t) this->screenDepth/8*this->screenWidth;
    }

    ScreenFormat DoomController::getScreenFormat() {
        if (this->doomRunning) return (ScreenFormat) this->gameVariables->SCREEN_FORMAT;
        else return this->screenFormat;
    }

    size_t DoomController::getScreenSize() {
        if (this->doomRunning) return (size_t) this->gameVariables->SCREEN_SIZE;
        else return (size_t) this->screenChannels * this->screenWidth * this->screenHeight;
    }

//SM SETTERS & GETTERS

    uint8_t *const DoomController::getScreen() { return this->screen; }

    DoomController::InputStruct *const DoomController::getInput() { return this->input; }

    DoomController::GameVariablesStruct *const DoomController::getGameVariables() { return this->gameVariables; }

    int DoomController::getButtonState(Button button){
        if(this->doomRunning) return this->input->BT[button];
        else return 0;
    }

    void DoomController::setButtonState(Button button, int state) {
        if (button < ButtonsNumber && button >= 0 && this->doomRunning)
            this->input->BT[button] = state;

    }

    void DoomController::toggleButtonState(Button button) {
        if (button < ButtonsNumber && button >= 0 && this->doomRunning)
            this->input->BT[button] = !this->input->BT[button];

    }

    bool DoomController::isButtonAvailable(Button button){
        if(this->doomRunning) return this->input->BT_AVAILABLE[button];
        else return this->_input->BT_AVAILABLE[button];
    }

    void DoomController::setButtonAvailable(Button button, bool allow) {
        if (button < ButtonsNumber && button >= 0) {
            if (this->doomRunning) this->input->BT_AVAILABLE[button] = allow;
            this->_input->BT_AVAILABLE[button] = allow;
        }
    }

    void DoomController::resetButtons(){
        if (this->doomRunning)
            for (int i = 0; i < ButtonsNumber; ++i)
                this->input->BT[i] = 0;
    }

    void DoomController::resetDescreteButtons(){
        if (this->doomRunning) {
            this->input->BT[ATTACK] = 0;
            this->input->BT[USE] = 0;

            this->input->BT[JUMP] = 0;
            this->input->BT[TURN180] = 0;
            this->input->BT[ALTATTACK] = 0;
            this->input->BT[RELOAD] = 0;
            this->input->BT[LAND] = 0;

            for (int i = SELECT_WEAPON1; i <= SELECT_WEAPON0; ++i) {
                this->input->BT[i] = 0;
            }

            this->input->BT[SELECT_NEXT_WEAPON] = 0;
            this->input->BT[SELECT_PREV_WEAPON] = 0;
            this->input->BT[DROP_SELECTED_WEAPON] = 0;
            this->input->BT[ACTIVATE_SELECTED_ITEM] = 0;
            this->input->BT[SELECT_NEXT_ITEM] = 0;
            this->input->BT[SELECT_PREV_ITEM] = 0;
            this->input->BT[DROP_SELECTED_ITEM] = 0;
        }
    }

    void DoomController::disableAllButtons(){
        for (int i = 0; i < ButtonsNumber; ++i){
            if (this->doomRunning) this->input->BT_AVAILABLE[i] = false;
            this->_input->BT_AVAILABLE[i] = false;
        }
    }

    void DoomController::availableAllButtons(){
        for (int i = 0; i < ButtonsNumber; ++i){
            if (this->doomRunning) this->input->BT_AVAILABLE[i] = true;
            this->_input->BT_AVAILABLE[i] = true;
        }
    }

    void DoomController::setButtonMaxValue(Button button, int value){
        if(button >= DiscreteButtonsNumber){
            if (this->doomRunning) this->input->BT_MAX_VALUE[button - DiscreteButtonsNumber] = value;
            this->_input->BT_MAX_VALUE[button - DiscreteButtonsNumber] = value;
        }
    }

    int DoomController::getButtonMaxValue(Button button){
        if(button >= DiscreteButtonsNumber){
            if (this->doomRunning) return this->input->BT_MAX_VALUE[button - DiscreteButtonsNumber];
            else return this->_input->BT_MAX_VALUE[button - DiscreteButtonsNumber];
        }
        else return 1;
    }

    bool DoomController::isButtonDiscrete(Button button){
        return button < DiscreteButtonsNumber;
    }

    bool DoomController::isButtonAxis(Button button){
        return button >= DiscreteButtonsNumber;
    }

    bool DoomController::isAllowDoomInput(){ return this->allowDoomInput; }
    void DoomController::setAllowDoomInput(bool set){ if(!this->doomRunning) this->allowDoomInput = set; }

    bool DoomController::isRunDoomAsync(){ return this->runDoomAsync; }
    void DoomController::setRunDoomAsync(bool set){ if(!this->doomRunning) this->runDoomAsync = set; }

    int DoomController::getGameVariable(GameVariable var) {
        switch (var) {
            case KILLCOUNT :
                return this->gameVariables->MAP_KILLCOUNT;
            case ITEMCOUNT :
                return this->gameVariables->MAP_ITEMCOUNT;
            case SECRETCOUNT :
                return this->gameVariables->MAP_SECRETCOUNT;
            case FRAGCOUNT:
                return this->gameVariables->PLAYER_FRAGCOUNT;
            case HEALTH :
                return this->gameVariables->PLAYER_HEALTH;
            case ARMOR :
                return this->gameVariables->PLAYER_ARMOR;
            case DEAD :
                return this->gameVariables->PLAYER_DEAD;
            case ON_GROUND :
                return this->gameVariables->PLAYER_ON_GROUND;
            case ATTACK_READY :
                return this->gameVariables->PLAYER_ATTACK_READY;
            case ALTATTACK_READY :
                return this->gameVariables->PLAYER_ALTATTACK_READY;
            case SELECTED_WEAPON :
                return this->gameVariables->PLAYER_SELECTED_WEAPON;
            case SELECTED_WEAPON_AMMO :
                return this->gameVariables->PLAYER_SELECTED_WEAPON_AMMO;
        }
        if(var >= AMMO0 && var <= AMMO9){
            return this->gameVariables->PLAYER_AMMO[var - AMMO0];
        }
        else if(var >= WEAPON0 && var <= WEAPON9){
            return this->gameVariables->PLAYER_WEAPON[var - WEAPON0];
        }
        else if(var >= USER1 && var <= USER30){
            return this->gameVariables->MAP_USER_VARS[var - USER1];
        }
        else return 0;
    }

    int DoomController::getGameTic() { return this->gameVariables->GAME_TIC; }
    int DoomController::getMapTic() { return this->gameVariables->MAP_TIC; }

    int DoomController::getMapReward() { return this->gameVariables->MAP_REWARD; }

    int DoomController::getMapKillCount() { return this->gameVariables->MAP_KILLCOUNT; }
    int DoomController::getMapItemCount() { return this->gameVariables->MAP_ITEMCOUNT; }
    int DoomController::getMapSecretCount() { return this->gameVariables->MAP_SECRETCOUNT; }

    bool DoomController::isPlayerDead() { return this->gameVariables->PLAYER_DEAD; }

    int DoomController::getPlayerKillCount() { return this->gameVariables->PLAYER_KILLCOUNT; }
    int DoomController::getPlayerItemCount() { return this->gameVariables->PLAYER_ITEMCOUNT; }
    int DoomController::getPlayerSecretCount() { return this->gameVariables->PLAYER_SECRETCOUNT; }
    int DoomController::getPlayerFragCount() { return this->gameVariables->PLAYER_FRAGCOUNT; }

    int DoomController::getPlayerHealth() { return this->gameVariables->PLAYER_HEALTH; }
    int DoomController::getPlayerArmor() { return this->gameVariables->PLAYER_ARMOR; }

    bool DoomController::isPlayerOnGround() { return this->gameVariables->PLAYER_ON_GROUND; }
    bool DoomController::isPlayerAttackReady() { return this->gameVariables->PLAYER_ATTACK_READY; }
    bool DoomController::isPlayerAltAttackReady() { return this->gameVariables->PLAYER_ALTATTACK_READY; }

    int DoomController::getPlayerSelectedWeaponAmmo() { return this->gameVariables->PLAYER_SELECTED_WEAPON_AMMO; }
    int DoomController::getPlayerSelectedWeapon() { return this->gameVariables->PLAYER_SELECTED_WEAPON; }

    int DoomController::getPlayerAmmo(unsigned int slot) {
        return slot < SlotsNumber ? this->gameVariables->PLAYER_AMMO[slot] : 0;
    }

    int DoomController::getPlayerWeapon(unsigned int slot) {
        return slot < SlotsNumber ? this->gameVariables->PLAYER_WEAPON[slot] : 0;
    }

//PRIVATE

    void DoomController::generateStaticSeed(){
        srand(time(NULL));
        this->staticSeed = rand();
    }

    void DoomController::generateInstanceId() {
        std::string chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";
        this->instanceId = "";

        srand(time(NULL));
        for (int i = 0; i < 10; ++i) {
            this->instanceId += chars[rand() % (chars.length() - 1)];
        }
    }

    void DoomController::waitForDoomStart() {

        this->doomWorking = true;

        MessageCommandStruct msg;

        unsigned int priority;
        bip::message_queue::size_type recv_size;

        this->MQControllerRecv(&msg, recv_size, priority);
        switch (msg.code) {
            case MSG_CODE_DOOM_DONE :
                this->doomRunning = true;
                break;

            case MSG_CODE_DOOM_CLOSE :
            case MSG_CODE_DOOM_PROCESS_EXIT :
                throw DoomUnexpectedExitException();

            case MSG_CODE_DOOM_ERROR :
                throw DoomErrorException();

            case MSG_CODE_SIGNAL_INT_ABRT_TERM :
                this->close();
                break;
        }

        this->doomWorking = false;
    }

    void DoomController::waitForDoomWork() {

        if(doomRunning){
            this->doomWorking = true;

            MessageCommandStruct msg;

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
                        this->close();
                        break;

                    case MSG_CODE_DOOM_ERROR :
                        throw DoomErrorException();

                    case MSG_CODE_DOOM_PROCESS_EXIT :
                        if(this->doomRunning) throw DoomUnexpectedExitException();
                        break;

                    case MSG_CODE_SIGNAL_INT_ABRT_TERM :
                        this->close();
                        exit(0);
                }
            } while (!done);

            this->doomWorking = false;
        }
        else throw DoomIsNotRunningException();
    }

    void DoomController::waitForDoomMapStartTime() {
        while(this->gameVariables->MAP_TIC < this->mapStartTime) {
            this->MQDoomSend(MSG_CODE_TIC);
            this->waitForDoomWork();
        }
    }

    void DoomController::launchDoom() {

        std::vector <std::string> args;

        //exe
        args.push_back(gamePath);

        //main wad
        args.push_back("-iwad");
        args.push_back(this->iwadPath);

        //skill
        args.push_back("-skill");
        args.push_back(b::lexical_cast<std::string>(this->skill));

        //wads
        if (this->filePath.length() != 0) {
            args.push_back("-file");
            args.push_back(this->filePath);
        }

        if (this->configPath.length() != 0) {
            args.push_back("-config");
            args.push_back(this->configPath);
        }

        if(this->useStaticSeed) {
            args.push_back("-rngseed");
            args.push_back(b::lexical_cast<std::string>(this->staticSeed));
        }

        //map
        args.push_back("+map");
        args.push_back(this->map);

        //resolution

        args.push_back("-width");
        //args.push_back("+vid_defwidth");
        args.push_back(b::lexical_cast<std::string>(this->screenWidth));

        args.push_back("-height");
        //args.push_back("+vid_defheight");
        args.push_back(b::lexical_cast<std::string>(this->screenHeight));

        //hud
        args.push_back("+screenblocks");
        if (this->hud) args.push_back("10");
        else args.push_back("12");

        //weapon
        args.push_back("+r_drawplayersprites");
        if (this->weapon) args.push_back("1");
        else args.push_back("0");

        //crosshair
        args.push_back("+crosshair");
        if (this->crosshair) {
            args.push_back("1");
            args.push_back("+crosshairhealth");
            args.push_back("0");
        }
        else args.push_back("0");

        //decals
        args.push_back("+cl_maxdecals");
        if (this->decals) args.push_back("1024");
        else args.push_back("0");

        //particles
        args.push_back("+r_particles");
        if (this->decals) args.push_back("1");
        else args.push_back("0");

        //weapon auto switch
        //args.push_back("+neverswitchonpickup");
        //args.push_back("1");

        //vizia args
        args.push_back("+vizia_controlled");
        args.push_back("1");

        args.push_back("+vizia_instance_id");
        args.push_back(this->instanceId);

        if(this->noConsole){
            args.push_back("+vizia_no_console");
            args.push_back("1");
        }

        if(this->allowDoomInput){
            args.push_back("+vizia_allow_input");
            args.push_back("1");

            //allow mouse
            args.push_back("+use_mouse");
            args.push_back("1");
        }
        else{
            //disable mouse
            args.push_back("+use_mouse");
            args.push_back("0");
        }

        if(this->runDoomAsync){
            args.push_back("+vizia_async");
            args.push_back("1");
        }

        args.push_back("+vizia_screen_format");
        args.push_back(b::lexical_cast<std::string>(this->screenFormat));

        args.push_back("+vizia_window_hidden");
        if (this->windowHidden) args.push_back("1");
        else args.push_back("0");

        args.push_back("+vizia_no_x_server");
        if (this->noXServer) args.push_back("1");
        else args.push_back("0");

        //no wipe animation
        args.push_back("+wipetype");
        args.push_back("0");

        //no sound/idle/joy
        args.push_back("-noidle");
        args.push_back("-nojoy");
        args.push_back("-nosound");

        //35 fps and no vsync
        args.push_back("+cl_capfps");
        args.push_back("1");

        args.push_back("+vid_vsync");
        args.push_back("0");

        //custom args
        for(int i = 0; i < this->customArgs.size(); ++i){
            args.push_back(customArgs[i]);
        }

        //bpr::context ctx;
        //ctx.stdout_behavior = bpr::silence_stream();
        bpr::child doomProcess = bpr::execute(bpri::set_args(args), bpri::inherit_env());
        bpr::wait_for_exit(doomProcess);
        this->MQControllerSend(MSG_CODE_DOOM_PROCESS_EXIT);
    }

    void DoomController::handleSignals(){
        ba::signal_set signals(this->ioService, SIGINT, SIGABRT, SIGTERM);
        signals.async_wait(b::bind(signalHandler, b::ref(signals), this, _1, _2));

        this->ioService.run();
    }

//SM FUNCTIONS 
    void DoomController::SMInit() {
        this->SMName = SM_NAME_BASE + instanceId;
        //bip::shared_memory_object::remove(this->SMName.c_str());
        try {
            this->SM = bip::shared_memory_object(bip::open_only, this->SMName.c_str(), bip::read_write);

            this->InputSMRegion = new bip::mapped_region(this->SM, bip::read_write, 0,
                                                         sizeof(DoomController::InputStruct));
            this->input = static_cast<DoomController::InputStruct *>(this->InputSMRegion->get_address());

            this->GameVariablesSMRegion = new bip::mapped_region(this->SM, bip::read_only,
                                                            sizeof(DoomController::InputStruct),
                                                            sizeof(DoomController::GameVariablesStruct));
            this->gameVariables = static_cast<DoomController::GameVariablesStruct *>(this->GameVariablesSMRegion->get_address());

            this->screenWidth = this->gameVariables->SCREEN_WIDTH;
            this->screenHeight = this->gameVariables->SCREEN_HEIGHT;
            this->screenPitch = this->gameVariables->SCREEN_PITCH;
            this->screenSize = this->gameVariables->SCREEN_SIZE;
            this->screenFormat = (ScreenFormat)this->gameVariables->SCREEN_FORMAT;

            this->ScreenSMRegion = new bip::mapped_region(this->SM, bip::read_only,
                                                          sizeof(DoomController::InputStruct) +
                                                          sizeof(DoomController::GameVariablesStruct),
                                                          this->screenSize);
            this->screen = static_cast<uint8_t *>(this->ScreenSMRegion->get_address());
        }
        catch (bip::interprocess_exception &ex) {
            throw SharedMemoryException();
        }
    }

    void DoomController::SMClose() {
        delete this->InputSMRegion;
        this->InputSMRegion = NULL;
        delete this->GameVariablesSMRegion;
        this->GameVariablesSMRegion = NULL;
        delete this->ScreenSMRegion;
        this->ScreenSMRegion = NULL;
        bip::shared_memory_object::remove(this->SMName.c_str());
    }

//MQ FUNCTIONS
    void DoomController::MQInit() {

        this->MQControllerName = MQ_NAME_CTR_BASE + instanceId;
        this->MQDoomName = MQ_NAME_DOOM_BASE + instanceId;

        try {
            bip::message_queue::remove(this->MQControllerName.c_str());
            bip::message_queue::remove(this->MQDoomName.c_str());

            this->MQController = new bip::message_queue(bip::open_or_create, this->MQControllerName.c_str(), MQ_MAX_MSG_NUM, MQ_MAX_MSG_SIZE);
            this->MQDoom = new bip::message_queue(bip::open_or_create, this->MQDoomName.c_str(), MQ_MAX_MSG_NUM, MQ_MAX_MSG_SIZE);
        }
        catch (bip::interprocess_exception &ex) {
            throw MessageQueueException();
        }
    }

    void DoomController::MQControllerSend(uint8_t code) {
        MessageSignalStruct msg;
        msg.code = code;
        this->MQController->send(&msg, sizeof(MessageSignalStruct), 0);
    }

    void DoomController::MQDoomSend(uint8_t code) {
        MessageSignalStruct msg;
        msg.code = code;
        this->MQDoom->send(&msg, sizeof(MessageSignalStruct), 0);
    }

    void DoomController::MQDoomSend(uint8_t code, const char *command) {
        MessageCommandStruct msg;
        msg.code = code;
        strncpy(msg.command, command, MQ_MAX_CMD_LEN);
        this->MQDoom->send(&msg, sizeof(MessageCommandStruct), 0);
    }

    void DoomController::MQControllerRecv(void *msg, unsigned long &size, unsigned int &priority) {
        this->MQController->receive(msg, sizeof(MessageCommandStruct), size, priority);
    }

    void DoomController::MQClose() {
        bip::message_queue::remove(this->MQDoomName.c_str());
        delete this->MQDoom;
        this->MQDoom = NULL;

        bip::message_queue::remove(this->MQControllerName.c_str());
        delete this->MQController;
        this->MQController = NULL;
    }
}
