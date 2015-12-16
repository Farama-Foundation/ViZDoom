#include "ViziaDoomController.h"

#include <vector>
#include <iostream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <cstdio>

namespace Vizia {

//PUBLIC FUNCTIONS

    DoomController::DoomController() {

        this->MQController = NULL;
        this->MQDoom = NULL;

        this->InputSMRegion = NULL;
        this->GameVarsSMRegion = NULL;
        this->ScreenSMRegion = NULL;

        this->screenWidth = 320;
        this->screenHeight = 240;
        this->screenChannels = 3;
        this->screenPitch = 0;
        this->screenSize = 0;
        this->screenFormat = CRCGCB;

        this->gamePath = "viziazdoom";
        this->iwadPath = "doom2.wad";
        this->filePath = "";
        this->map = "map01";
        this->configPath = "";
        this->skill = 1;

        this->hud = true;
        this->weapon = true;
        this->crosshair = false;
        this->decals = true;
        this->particles = true;

        this->windowHidden = false;
        this->noXServer = false;

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

        // AUTO RESTART

        this->generateInstanceId();
        this->generateSeed();
        this->doomRunning = false;
        this->doomTic = false;
    }

    DoomController::~DoomController() {
        this->close();
    }

//FLOW CONTROL

    bool DoomController::init() {

        if (!this->doomRunning && this->iwadPath.length() != 0 && this->map.length() != 0) {

            try{
                if (this->instanceId.length() == 0) generateInstanceId();

                this->MQInit();
                doomThread = new b::thread(b::bind(&DoomController::lunchDoom, this));
                this->waitForDoomStart();

                this->doomRunning = true;

                this->SMInit();

                this->waitForDoomMapStartTime();
            }
            catch(const Exception &e){
                this->close();
                throw;
            }
        }

        return this->doomRunning;
    }

    void DoomController::close() {

        if (this->doomRunning) {
            this->MQSend(MSG_CODE_CLOSE);

            if (this->doomThread->joinable()) {
                this->doomThread->interrupt();
                this->doomThread->join();
            }

            this->doomRunning = false;
        }

        this->SMClose();
        this->MQClose();
    }

    void DoomController::restart() {
        close();
        init();
    }

    bool DoomController::tic() {

        if (doomRunning) {

            if (!this->mapEnded) {
                this->mapLastTic = this->GameVars->MAP_TIC;
                this->waitForDoomTic();
            }

            if (this->GameVars->PLAYER_DEAD) {
                this->mapEnded = true;
                if (this->autoRestart && this->autoRestartOnPlayersDeath) this->restartMap();
            }
            else if (this->mapTimeout > 0 && this->GameVars->MAP_TIC >= this->mapTimeout + this->mapStartTime) {
                this->mapEnded = true;
                if (this->autoRestart && this->autoRestartOnTimeout) this->restartMap();
            }
            else if (this->GameVars->MAP_END) {
                this->mapEnded = true;
                if (this->autoRestart && this->autoRestartOnMapEnd) this->restartMap();
            }
        }

        return true;
    }

    void DoomController::restartMap() {
        this->setMap(this->map);
    }

    void DoomController::resetMap() {
        this->restartMap();
    }

    void DoomController::sendCommand(std::string command) {
        this->MQSend(MSG_CODE_COMMAND, command.c_str());
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
            this->sendCommand("map " + this->map);
            if (map != this->map) this->mapRestartCount = 0;
            else ++this->mapRestartCount;

            this->mapRestarting = true;

            this->resetInput();

            int restartingTics = 0;

            do {
                ++restartingTics;
                this->waitForDoomTic();

                if (restartingTics > 4) {
                    this->sendCommand("map " + this->map);
                    restartingTics = 0;
                }

            } while (this->GameVars->MAP_END || this->GameVars->MAP_TIC > 1);

            this->waitForDoomMapStartTime();

            this->mapRestarting = false;
            this->mapEnded = false;
        }
    }

    int DoomController::getSkill(){ return this->skill; }
    void DoomController::setSkill(int skill) {
        this->skill = skill;
        if (this->doomRunning) {
            this->sendCommand("skill " + this->skill);
            //this->resetMap();
        }
    }

    unsigned int DoomController::getCurrentSeed(){ return this->GameVars->GAME_SEED; }
    unsigned int DoomController::getSeed(){ return this->seed; }
    void DoomController::setSeed(unsigned int seed){ if(!this->doomRunning) this->seed = seed; }

    void DoomController::setAutoMapRestart(bool set) { this->autoRestart = set; }
    void DoomController::setAutoMapRestartOnTimeout(bool set) { this->autoRestartOnTimeout = set; }
    void DoomController::setAutoMapRestartOnPlayerDeath(bool set) { this->autoRestartOnPlayersDeath = set; }
    void DoomController::setAutoMapRestartOnMapEnd(bool set) { this->autoRestartOnMapEnd = set; }

    unsigned int DoomController::getMapStartTime() { return this->mapStartTime; }
    void DoomController::setMapStartTime(unsigned int tics) { this->mapStartTime = tics ? tics : 1; }

    unsigned int DoomController::getMapTimeout() { return this->mapTimeout; }
    void DoomController::setMapTimeout(unsigned int tics) { this->mapTimeout = tics; }

    bool DoomController::isMapFirstTic() {
        if (this->doomRunning && this->GameVars->MAP_TIC <= 1) return true;
        else return false;
    }

    bool DoomController::isMapLastTic() {
        if (this->doomRunning && this->mapTimeout > 0 && this->GameVars->MAP_TIC >= this->mapTimeout + this->mapStartTime) return true;
        else return false;
    }

    bool DoomController::isMapEnded() {
        if (this->doomRunning && this->GameVars->MAP_END) return true;
        else return false;
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
                case CRCGCBCA:
                case RGBA32:
                case ARGB32:
                case CBCGCRCA:
                case BGRA32:
                case ABGR32:
                    this->screenChannels = 4;
                    break;
                case GRAY8:
                case DOOM_256_COLORS:
                    this->screenChannels = 1;
                    break;
                default:
                    this->screenChannels = 0;
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
            if (this->decals) this->sendCommand("cl_maxdecals 128");
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

    int DoomController::getScreenWidth() {
        if (this->doomRunning) return this->GameVars->SCREEN_WIDTH;
        else return 0;
    }

    int DoomController::getScreenHeight() {
        if (this->doomRunning) return this->GameVars->SCREEN_HEIGHT;
        else return 0;
    }
    int DoomController::getScreenChannels() {
        if (this->doomRunning) return this->screenChannels;
        else return 0;
    }

    size_t DoomController::getScreenPitch() {
        if (this->doomRunning) return (size_t) this->GameVars->SCREEN_PITCH;
        else return 0;
    }

    ScreenFormat DoomController::getScreenFormat() {
        if (this->doomRunning) return (ScreenFormat) this->GameVars->SCREEN_FORMAT;
        else return this->screenFormat;
    }

    size_t DoomController::getScreenSize() {
        if (this->doomRunning) return (size_t) this->GameVars->SCREEN_SIZE;
        else return 0;
    }

//SM SETTERS & GETTERS

    uint8_t *const DoomController::getScreen() { return this->Screen; }

    DoomController::InputStruct *const DoomController::getInput() { return this->Input; }

    DoomController::GameVarsStruct *const DoomController::getGameVars() { return this->GameVars; }

    void DoomController::setMouse(int x, int y) {
        this->Input->MS_X = x;
        this->Input->MS_Y = y;
    }

    int DoomController::getMouseX() { return this->Input->MS_X; }
    void DoomController::setMouseX(int x) { this->Input->MS_X = x; }

    int DoomController::getMouseY() { return this->Input->MS_Y; }
    void DoomController::setMouseY(int y) { this->Input->MS_Y = y; }

    void DoomController::resetMouse(){
        this->Input->MS_X = 0;
        this->Input->MS_Y = 0;
    }

    bool DoomController::getButtonState(Button button){ return this->Input->BT[button]; }

    void DoomController::setButtonState(Button button, bool state) {
        if (button < ButtonsNumber && button >= 0) this->Input->BT[button] = state;
    }

    void DoomController::toggleButtonState(Button button) {
        if (button < ButtonsNumber && button >= 0) this->Input->BT[button] = !this->Input->BT[button];
    }

    bool DoomController::isButtonAvailable(Button button){ return this->Input->BT_AVAILABLE[button]; }

    void DoomController::setButtonAvailable(Button button, bool allow) {
        if (button < ButtonsNumber && button >= 0) this->Input->BT_AVAILABLE[button] = allow;
    }

    void DoomController::resetButtons(){
        for (int i = 0; i < ButtonsNumber; ++i) this->Input->BT[i] = false;
    }

    void DoomController::resetDescreteButtons(){
        this->Input->BT[ATTACK] = false;
        this->Input->BT[USE] = false;

        this->Input->BT[JUMP] = false;
        this->Input->BT[TURN180] = false;
        this->Input->BT[ALTATTACK] = false;
        this->Input->BT[RELOAD] = false;

        for(int i = SELECT_WEAPON1; i <= SELECT_WEAPON7; ++i){
            this->Input->BT[i] = false;
        }

        this->Input->BT[SELECT_NEXT_WEAPON] = false;
        this->Input->BT[SELECT_PREV_WEAPON] = false;
    }

    void DoomController::disableAllButtons(){
        for (int i = 0; i < ButtonsNumber; ++i) this->Input->BT_AVAILABLE[i] = false;
    }

    void DoomController::availableAllButtons(){
        for (int i = 0; i < ButtonsNumber; ++i) this->Input->BT_AVAILABLE[i] = true;
    }

    void DoomController::resetInput() {
        this->resetMouse();
        this->resetButtons();
    }

    bool DoomController::isAllowDoomInput(){ return this->allowDoomInput; }
    void DoomController::setAllowDoomInput(bool set){ if(!this->doomRunning) this->allowDoomInput = set; }

    int DoomController::getGameVar(GameVar var) {
        switch (var) {
            case KILLCOUNT :
                return this->GameVars->MAP_KILLCOUNT;
            case ITEMCOUNT :
                return this->GameVars->MAP_ITEMCOUNT;
            case SECRETCOUNT :
                return this->GameVars->MAP_SECRETCOUNT;
            case HEALTH :
                return this->GameVars->PLAYER_HEALTH;
            case ARMOR :
                return this->GameVars->PLAYER_ARMOR;
            case SELECTED_WEAPON :
                return this->GameVars->PLAYER_SELECTED_WEAPON;
            case SELECTED_WEAPON_AMMO :
                return this->GameVars->PLAYER_SELECTED_WEAPON_AMMO;
        }
        if(var >= AMMO1 && var <= AMMO4){
            return this->GameVars->PLAYER_AMMO[var-AMMO1];
        }
        else if(var >= WEAPON1 && var <= WEAPON7){
            return this->GameVars->PLAYER_WEAPON[var-WEAPON1];
        }
        else if(var >= KEY1 && var <= KEY3){
            return this->GameVars->PLAYER_WEAPON[var-KEY1];
        }
        else if(var >= USER1 && var <= USER30){
            return this->GameVars->MAP_USER_VARS[var-USER1];
        }
        else return 0;
    }

    int DoomController::getGameTic() { return this->GameVars->GAME_TIC; }
    int DoomController::getMapTic() { return this->GameVars->MAP_TIC; }

    int DoomController::getMapReward() { return this->GameVars->MAP_REWARD; }

    int DoomController::getMapKillCount() { return this->GameVars->MAP_KILLCOUNT; }
    int DoomController::getMapItemCount() { return this->GameVars->MAP_ITEMCOUNT; }
    int DoomController::getMapSecretCount() { return this->GameVars->MAP_SECRETCOUNT; }

    bool DoomController::isPlayerDead() { return this->GameVars->PLAYER_DEAD; }

    int DoomController::getPlayerKillCount() { return this->GameVars->PLAYER_KILLCOUNT; }
    int DoomController::getPlayerItemCount() { return this->GameVars->PLAYER_ITEMCOUNT; }
    int DoomController::getPlayerSecretCount() { return this->GameVars->PLAYER_SECRETCOUNT; }
    int DoomController::getPlayerFragCount() { return this->GameVars->PLAYER_FRAGCOUNT; }

    int DoomController::getPlayerHealth() { return this->GameVars->PLAYER_HEALTH; }
    int DoomController::getPlayerArmor() { return this->GameVars->PLAYER_ARMOR; }

    int DoomController::getPlayerSelectedWeaponAmmo() { return this->GameVars->PLAYER_SELECTED_WEAPON_AMMO; }
    int DoomController::getPlayerSelectedWeapon() { return this->GameVars->PLAYER_SELECTED_WEAPON; }

    int DoomController::getPlayerAmmo1() { return this->GameVars->PLAYER_AMMO[0]; }
    int DoomController::getPlayerAmmo2() { return this->GameVars->PLAYER_AMMO[1]; }
    int DoomController::getPlayerAmmo3() { return this->GameVars->PLAYER_AMMO[2]; }
    int DoomController::getPlayerAmmo4() { return this->GameVars->PLAYER_AMMO[3]; }

    bool DoomController::getPlayerWeapon1() { return this->GameVars->PLAYER_WEAPON[0]; }
    bool DoomController::getPlayerWeapon2() { return this->GameVars->PLAYER_WEAPON[1]; }
    bool DoomController::getPlayerWeapon3() { return this->GameVars->PLAYER_WEAPON[2]; }
    bool DoomController::getPlayerWeapon4() { return this->GameVars->PLAYER_WEAPON[3]; }
    bool DoomController::getPlayerWeapon5() { return this->GameVars->PLAYER_WEAPON[4]; }
    bool DoomController::getPlayerWeapon6() { return this->GameVars->PLAYER_WEAPON[5]; }
    bool DoomController::getPlayerWeapon7() { return this->GameVars->PLAYER_WEAPON[6]; }

    bool DoomController::getPlayerKey1() { return this->GameVars->PLAYER_KEY[0]; }
    bool DoomController::getPlayerKey2() { return this->GameVars->PLAYER_KEY[1]; }
    bool DoomController::getPlayerKey3() { return this->GameVars->PLAYER_KEY[2]; }

//PRIVATE

    void DoomController::generateSeed(){
        srand(time(NULL));
        this->seed = rand();
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

        this->doomTic = true;

        MessageCommandStruct msg;

        unsigned int priority;
        bip::message_queue::size_type recvd_size;

        MQController->receive(&msg, sizeof(MessageCommandStruct), recvd_size, priority);

        switch (msg.code) {
            case MSG_CODE_DOOM_READY :
            case MSG_CODE_DOOM_TIC :
                this->doomRunning = true;
                break;

            case MSG_CODE_DOOM_CLOSE :
                throw DoomUnexpectedExitException();

            case MSG_CODE_DOOM_ERROR :
                throw DoomErrorException();

            default :
                break;
        }

        this->doomTic = false;
    }

    void DoomController::waitForDoomTic() {
        if (this->doomRunning) {

            this->MQSend(MSG_CODE_TIC);

            this->doomTic = true;

            MessageCommandStruct msg;

            unsigned int priority;
            bip::message_queue::size_type recvd_size;

            bool nextTic = false;
            do {
                MQController->receive(&msg, sizeof(MessageCommandStruct), recvd_size, priority);
                switch (msg.code) {
                    case MSG_CODE_DOOM_READY :
                    case MSG_CODE_DOOM_TIC :
                        nextTic = true;
                        break;

                    case MSG_CODE_DOOM_CLOSE :
                        this->close();
                        throw DoomUnexpectedExitException();

                    case MSG_CODE_DOOM_ERROR :
                        this->close();
                        throw DoomErrorException();

                    default :
                        break;
                }
            } while (!nextTic);

            this->doomTic = false;
        }
    }

    void DoomController::waitForDoomMapStartTime() {
        while(this->GameVars->MAP_TIC < this->mapStartTime) {
            this->waitForDoomTic();
        }
    }

    void DoomController::lunchDoom() {

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

        args.push_back("-rngseed");
        args.push_back(b::lexical_cast<std::string>(this->seed));

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
        if (this->decals) args.push_back("128");
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

        args.push_back("+vizia_singletic");
        args.push_back("1");

        if(this->allowDoomInput){
            args.push_back("+vizia_allow_input");
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

        //temp disable mouse
        args.push_back("+use_mouse");
        args.push_back("0");

        //35 fps and no vsync
        args.push_back("+cl_capfps");
        args.push_back("1");

        args.push_back("+vid_vsync");
        args.push_back("0");

        //bpr::context ctx;
        //ctx.stdout_behavior = bpr::silence_stream();
        bpr::child doomProcess = bpr::execute(bpri::set_args(args), bpri::inherit_env());
        bpr::wait_for_exit(doomProcess);
        this->MQSelfSend(MSG_CODE_DOOM_CLOSE);
    }

//SM FUNCTIONS 
    void DoomController::SMInit() {
        this->SMName = SM_NAME_BASE + instanceId;
        //bip::shared_memory_object::remove(this->SMName.c_str());
        try {
            this->SM = bip::shared_memory_object(bip::open_only, this->SMName.c_str(), bip::read_write);

            this->InputSMRegion = new bip::mapped_region(this->SM, bip::read_write, 0,
                                                         sizeof(DoomController::InputStruct));
            this->Input = static_cast<DoomController::InputStruct *>(this->InputSMRegion->get_address());

            this->GameVarsSMRegion = new bip::mapped_region(this->SM, bip::read_only,
                                                            sizeof(DoomController::InputStruct),
                                                            sizeof(DoomController::GameVarsStruct));
            this->GameVars = static_cast<DoomController::GameVarsStruct *>(this->GameVarsSMRegion->get_address());

            this->screenWidth = this->GameVars->SCREEN_WIDTH;
            this->screenHeight = this->GameVars->SCREEN_HEIGHT;
            this->screenPitch = this->GameVars->SCREEN_PITCH;
            this->screenSize = this->GameVars->SCREEN_SIZE;
            this->screenFormat = (ScreenFormat)this->GameVars->SCREEN_FORMAT;

            this->ScreenSMRegion = new bip::mapped_region(this->SM, bip::read_only,
                                                          sizeof(DoomController::InputStruct) +
                                                          sizeof(DoomController::GameVarsStruct),
                                                          this->screenSize);
            this->Screen = static_cast<uint8_t *>(this->ScreenSMRegion->get_address());
        }
        catch (bip::interprocess_exception &ex) {
            throw SharedMemoryException();
        }
    }

    void DoomController::SMClose() {
        delete (this->InputSMRegion);
        this->InputSMRegion = NULL;
        delete (this->GameVarsSMRegion);
        this->GameVarsSMRegion = NULL;
        delete (this->ScreenSMRegion);
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

    void DoomController::MQSend(uint8_t code) {
        MessageSignalStruct msg;
        msg.code = code;
        this->MQDoom->send(&msg, sizeof(MessageSignalStruct), 0);
    }

    void DoomController::MQSelfSend(uint8_t code) {
        MessageSignalStruct msg;
        msg.code = code;
        this->MQController->send(&msg, sizeof(MessageSignalStruct), 0);
    }

    bool DoomController::MQTrySend(uint8_t code) {
        MessageSignalStruct msg;
        msg.code = code;
        return this->MQDoom->try_send(&msg, sizeof(MessageSignalStruct), 0);
    }

    void DoomController::MQSend(uint8_t code, const char *command) {
        MessageCommandStruct msg;
        msg.code = code;
        strncpy(msg.command, command, MQ_MAX_CMD_LEN);
        this->MQDoom->send(&msg, sizeof(MessageCommandStruct), 0);
    }

    bool DoomController::MQTrySend(uint8_t code, const char *command) {
        MessageCommandStruct msg;
        msg.code = code;
        strncpy(msg.command, command, MQ_MAX_CMD_LEN);
        return this->MQDoom->try_send(&msg, sizeof(MessageCommandStruct), 0);
    }

    void DoomController::MQRecv(void *msg, unsigned long &size, unsigned int &priority) {
        this->MQController->receive(&msg, sizeof(MessageCommandStruct), size, priority);
    }

    bool DoomController::MQTryRecv(void *msg, unsigned long &size, unsigned int &priority) {
        return this->MQController->try_receive(&msg, sizeof(MessageCommandStruct), size, priority);
    }

    void DoomController::MQClose() {
        bip::message_queue::remove(this->MQDoomName.c_str());
        delete(this->MQDoom);
        this->MQDoom = NULL;

        bip::message_queue::remove(this->MQControllerName.c_str());
        delete(this->MQController);
        this->MQController = NULL;
    }
}
