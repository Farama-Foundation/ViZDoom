#include "ViziaDoomController.h"

#include <vector>
#include <iostream>
#include <string>
#include <cstdlib>
#include <ctime>

ViziaButton ViziaDoomController::getButtonId(std::string name){
    if(name.compare("ATTACK") == 0) return ATTACK;
    else if(name.compare("USE") == 0) return USE;
    else if(name.compare("JUMP") == 0) return JUMP;
    else if(name.compare("CROUCH") == 0) return CROUCH;
    else if(name.compare("TURN180") == 0) return TURN180;
    else if(name.compare("ALTATTACK") == 0) return ALTATTACK;
    else if(name.compare("RELOAD") == 0) return RELOAD;
    else if(name.compare("ZOOM") == 0) return ZOOM;
    else if(name.compare("SPEED") == 0) return SPEED;
    else if(name.compare("STRAFE") == 0) return STRAFE;
    else if(name.compare("MOVERIGHT") == 0) return MOVERIGHT;
    else if(name.compare("MOVELEFT") == 0) return MOVELEFT;
    else if(name.compare("BACK") == 0) return BACK;
    else if(name.compare("FORWARD") == 0) return FORWARD;
    else if(name.compare("RIGHT") == 0) return RIGHT;
    else if(name.compare("LEFT") == 0) return LEFT;
    else if(name.compare("LOOKUP") == 0) return LOOKUP;
    else if(name.compare("LOOKDOWN") == 0) return LOOKDOWN;
    else if(name.compare("MOVEUP") == 0) return MOVEUP;
    else if(name.compare("MOVEDOWN") == 0) return MOVEDOWN;
    else if(name.compare("WEAPON1") == 0) return SELECT_WEAPON1;
    else if(name.compare("WEAPON2") == 0) return SELECT_WEAPON2;
    else if(name.compare("WEAPON3") == 0) return SELECT_WEAPON3;
    else if(name.compare("WEAPON4") == 0) return SELECT_WEAPON4;
    else if(name.compare("WEAPON5") == 0) return SELECT_WEAPON5;
    else if(name.compare("WEAPON6") == 0) return SELECT_WEAPON6;
    else if(name.compare("WEAPON7") == 0) return SELECT_WEAPON7;
    else if(name.compare("WEAPONNEXT") == 0) return SELECT_NEXT_WEAPON;
    else if(name.compare("WEAPONPREV") == 0) return SELECT_PREV_WEAPON;
    else return UNDEFINED_BUTTON;
};

ViziaGameVar ViziaDoomController::getGameVarId(std::string name){
    if(name.compare("KILLCOUNT") == 0) return KILLCOUNT;
    else if(name.compare("ITEMCOUNT") == 0) return ITEMCOUNT;
    else if(name.compare("SECRETCOUNT") == 0) return SECRETCOUNT;
    else if(name.compare("HEALTH") == 0) return HEALTH;
    else if(name.compare("ARMOR") == 0) return ARMOR;
    else if(name.compare("SELECTED_WEAPON") == 0) return SELECTED_WEAPON;
    else if(name.compare("SELECTED_WEAPON_AMMO") == 0) return SELECTED_WEAPON_AMMO;
    else if(name.compare("AMMO1") == 0) return AMMO1;
    else if(name.compare("AMMO_CLIP") == 0) return AMMO1;
    else if(name.compare("AMMO2") == 0) return AMMO2;
    else if(name.compare("AMMO_SHELL") == 0) return AMMO2;
    else if(name.compare("AMMO3") == 0) return AMMO3;
    else if(name.compare("AMMO_ROCKET") == 0) return AMMO3;
    else if(name.compare("AMMO4") == 0) return AMMO4;
    else if(name.compare("AMMO_CELL") == 0) return AMMO4;
    else if(name.compare("WEAPON1") == 0) return WEAPON1;
    else if(name.compare("WEAPON_FIST") == 0) return WEAPON1;
    else if(name.compare("WEAPON_CHAINSAW") == 0) return WEAPON1;
    else if(name.compare("WEAPON2") == 0) return WEAPON2;
    else if(name.compare("WEAPON_PISTOL") == 0) return WEAPON2;
    else if(name.compare("WEAPON3") == 0) return WEAPON3;
    else if(name.compare("WEAPON_SHOTGUN") == 0) return WEAPON3;
    else if(name.compare("WEAPON_SSG") == 0) return WEAPON3;
    else if(name.compare("WEAPON_SUPER_SHOTGUN") == 0) return WEAPON3;
    else if(name.compare("WEAPON4") == 0) return WEAPON4;
    else if(name.compare("WEAPON_CHAINGUN") == 0) return WEAPON4;
    else if(name.compare("WEAPON5") == 0) return WEAPON5;
    else if(name.compare("WEAPON_ROCKET_LUNCHER") == 0) return WEAPON5;
    else if(name.compare("WEAPON6") == 0) return WEAPON6;
    else if(name.compare("WEAPON_PLASMA_GUN") == 0) return WEAPON6;
    else if(name.compare("WEAPON7") == 0) return WEAPON7;
    else if(name.compare("WEAPON_BFG") == 0) return WEAPON7;
    else if(name.compare("KEY1") == 0) return KEY1;
    else if(name.compare("KEY_BLUE") == 0) return KEY1;
    else if(name.compare("KEY2") == 0) return KEY2;
    else if(name.compare("KEY_RED") == 0) return KEY2;
    else if(name.compare("KEY3") == 0) return KEY3;
    else if(name.compare("KEY_YELLOW") == 0) return KEY3;
    else return UNDEFINED_VAR;
}

//PUBLIC FUNCTIONS

ViziaDoomController::ViziaDoomController(){
    this->screenWidth = 320;
    this->screenHeight = 240;
    this->screenPitch = 0;
    this->screenSize = 0;
    this->screenFormat = CRCGCB;

    this->gamePath = "zdoom";
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

    this->autoRestart = false;
    this->autoRestartOnTimeout = true;
    this->autoRestartOnPlayersDeath = true;
    this->autoRestartOnMapEnd = true;
    this->mapTimeout = 0;
    this->mapRestartCount = 0;
    this->mapRestarting = false;
    this->mapEnded = false;
    this->mapLastTic = 1;

    // AUTO RESTART

    generateInstanceId();
    this->doomRunning = false;
    this->doomTic = false;
}

ViziaDoomController::~ViziaDoomController(){
    this->close();
}

//FLOW CONTROL

bool ViziaDoomController::init(){

    if(!this->doomRunning && this->iwadPath.length() != 0 && this->map.length() != 0) {

        if(this->instanceId.length() == 0) generateInstanceId();

        this->MQInit();
        //this->SMInit();

        //this->lunchDoom();
        doomThread = new b::thread(b::bind(&ViziaDoomController::lunchDoom, this));
        this->waitForDoomStart();

        if(this->doomRunning) {
            this->SMInit();
            return true;
        }
        else{
            this->MQClose();
            return false;
        }
    }

    return false;
}

bool ViziaDoomController::close(){

//    if(this->doomRunning) {
//        bpr::terminate(this->doomProcess);
//    }

    if(this->doomRunning) {
        this->MQSend(VIZIA_MSG_CODE_CLOSE);

        doomThread->interrupt();
        doomThread->join();

        this->SMClose();
        this->MQClose();

        this->doomRunning = false;

        return true;
    }

    return false;
}

bool ViziaDoomController::tic(){

    if(doomRunning) {

        if(this->GameVars->PLAYER_DEAD){
            this->mapEnded = true;
            if(this->autoRestart && this->autoRestartOnPlayersDeath) this->restartMap();
        }
        else if(this->mapTimeout > 0 && this->GameVars->MAP_TIC >= mapTimeout){
            this->mapEnded = true;
            if(this->autoRestart && this->autoRestartOnTimeout) this->restartMap();
        }
        else if(this->GameVars->MAP_END){
            this->mapEnded = true;
            if(this->autoRestart && this->autoRestartOnMapEnd) this->restartMap();
        }

        if(!this->mapEnded) {

            this->mapLastTic = this->GameVars->MAP_TIC;

            this->waitForDoomTic();
        }
    }

    return true;
}

bool ViziaDoomController::update(){
    return this->tic();
}

void ViziaDoomController::restartMap(){
    this->setMap(this->map);
}

void ViziaDoomController::resetMap(){
    this->restartMap();
}

void ViziaDoomController::restartGame(){
    //TO DO
}

void ViziaDoomController::sendCommand(std::string command){
    this->MQSend(VIZIA_MSG_CODE_COMMAND, command.c_str());
}

bool ViziaDoomController::isDoomRunning(){ return this->doomRunning; }

//SETTINGS

//GAME & MAP SETTINGS

void ViziaDoomController::setInstanceId(std::string id){ this->instanceId = id; }
void ViziaDoomController::setGamePath(std::string path){ this->gamePath = path; }
void ViziaDoomController::setIwadPath(std::string path){ this->iwadPath = path; }
void ViziaDoomController::setFilePath(std::string path){ this->filePath = path; }
void ViziaDoomController::setConfigPath(std::string path){ this->configPath = path; }

void ViziaDoomController::setMap(std::string map){
    this->map = map;
    if(this->doomRunning && !this->mapRestarting) {
        this->sendCommand("map " + this->map);
        if(map != this->map) this->mapRestartCount = 0;
        else ++this->mapRestartCount;

        this->mapRestarting = true;

        this->resetInput();

        int restartingTics = 0;

        do{
            ++restartingTics;
            this->waitForDoomTic();

            if(restartingTics > 4){
                this->sendCommand("map " + this->map);
                restartingTics = 0;
            }

        }while(this->GameVars->MAP_END || this->GameVars->MAP_TIC > 1);

        this->mapRestarting = false;
        this->mapEnded = false;
    }
}

void ViziaDoomController::setSkill(int skill){
    this->skill = skill;
    if(this->doomRunning){
        this->sendCommand("skill "+this->skill);
        //this->resetMap();
    }
}

void ViziaDoomController::setAutoMapRestart(bool set){ this->autoRestart = set; }
void ViziaDoomController::setAutoMapRestartOnTimeout(bool set){ this->autoRestartOnTimeout = set; }
void ViziaDoomController::setAutoMapRestartOnPlayerDeath(bool set){ this->autoRestartOnPlayersDeath = set; }
void ViziaDoomController::setAutoMapRestartOnMapEnd(bool set){ this->autoRestartOnMapEnd = set; }
void ViziaDoomController::setMapTimeout(unsigned int tics){ this->mapTimeout = tics; }

bool ViziaDoomController::isMapFirstTic(){
    if(this->GameVars->MAP_TIC <= 1) return true;
    else return false;
}

bool ViziaDoomController::isMapLastTic(){
    if(this->mapTimeout > 0 && this->GameVars->MAP_TIC >= this->mapTimeout) return true;
    else return false;
}

bool ViziaDoomController::isMapEnded(){
    if(this->GameVars->MAP_END) return true;
    else return false;
}

void ViziaDoomController::setScreenResolution(int width, int height){
    this->screenWidth = width;
    this->screenHeight = height;
}

void ViziaDoomController::setScreenWidth(int width){ this->screenWidth = width; }
void ViziaDoomController::setScreenHeight(int height){ this->screenHeight = height; }
void ViziaDoomController::setScreenFormat(ViziaScreenFormat format){ this->screenFormat = format; }

void ViziaDoomController::setRenderHud(bool hud){
    this->hud = hud;
    if(this->doomRunning){
        if(this->hud) this->sendCommand("screenblocks 10");
        else this->sendCommand("screenblocks 12");
    }
}

void ViziaDoomController::setRenderWeapon(bool weapon){
    this->weapon = weapon;
    if(this->doomRunning){
        if(this->weapon) this->sendCommand("r_drawplayersprites 1");
        else this->sendCommand("r_drawplayersprites 1");
    }
}

void ViziaDoomController::setRenderCrosshair(bool crosshair){
    this->crosshair = crosshair;
    if(this->doomRunning){
        if(this->crosshair){
            this->sendCommand("crosshairhealth false");
            this->sendCommand("crosshair 1");
        }
        else this->sendCommand("crosshair 0");
    }
}

void ViziaDoomController::setRenderDecals(bool decals){
    this->decals = decals;
    if(this->doomRunning){
        if(this->decals) this->sendCommand("cl_maxdecals 128");
        else this->sendCommand("cl_maxdecals 0");
    }
}

void ViziaDoomController::setRenderParticles(bool particles){
    this->particles = particles;
    if(this->doomRunning){
        if(this->particles) this->sendCommand("r_particles 1");
        else this->sendCommand("r_particles 0");
    }
}

int ViziaDoomController::getScreenWidth(){
    if(this->doomRunning) return this->GameVars->SCREEN_WIDTH;
    else return 0;
}

int ViziaDoomController::getScreenHeight(){
    if(this->doomRunning) return this->GameVars->SCREEN_HEIGHT;
    else return 0;
}

size_t ViziaDoomController::getScreenPitch(){
    if(this->doomRunning) return (size_t)this->GameVars->SCREEN_PITCH;
    else return 0;
}

ViziaScreenFormat ViziaDoomController::getScreenFormat(){
    if(this->doomRunning) return (ViziaScreenFormat)this->GameVars->SCREEN_FORMAT;
    else return this->screenFormat;
}

size_t ViziaDoomController::getScreenSize(){
    if(this->doomRunning) return (size_t)this->GameVars->SCREEN_SIZE;
    else return 0;
}

//SM SETTERS & GETTERS

uint8_t* const ViziaDoomController::getScreen() { return this->Screen; }
ViziaDoomController::InputStruct* const ViziaDoomController::getInput() { return this->Input; }
ViziaDoomController::GameVarsStruct* const ViziaDoomController::getGameVars() { return this->GameVars; }

void ViziaDoomController::setMouse(int x, int y){
    this->Input->MS_X = x;
    this->Input->MS_Y = y;
}

void ViziaDoomController::setMouseX(int x){
    this->Input->MS_X = x;
}

void ViziaDoomController::setMouseY(int y){
    this->Input->MS_Y = y;
}

void ViziaDoomController::setButtonState(ViziaButton button, bool state){
    if( button < ViziaButtonsNumber && button >= 0 ) this->Input->BT[button] = state;
}

void ViziaDoomController::toggleButtonState(ViziaButton button){
    if( button < ViziaButtonsNumber && button >= 0 ) this->Input->BT[button] = !this->Input->BT[button];
}

void ViziaDoomController::setAllowButton(ViziaButton button, bool allow){
    if( button < ViziaButtonsNumber && button >= 0 ) this->Input->BT_AVAILABLE[button] = allow;
}

void ViziaDoomController::resetInput(){
    this->Input->MS_X = 0;
    this->Input->MS_Y = 0;
    for(int i =0; i < ViziaButtonsNumber; ++i) this->Input->BT[i] = false;
}

int ViziaDoomController::getGameVar(ViziaGameVar var){
    switch(var){
        case KILLCOUNT : return this->GameVars->MAP_KILLCOUNT;
        case ITEMCOUNT : return this->GameVars->MAP_ITEMCOUNT;
        case SECRETCOUNT : return this->GameVars->MAP_SECRETCOUNT;
        case HEALTH : return this->GameVars->PLAYER_HEALTH;
        case ARMOR : return this->GameVars->PLAYER_ARMOR;
        case SELECTED_WEAPON : return this->GameVars->PLAYER_SELECTED_WEAPON;
        case SELECTED_WEAPON_AMMO : return this->GameVars->PLAYER_SELECTED_WEAPON_AMMO;
        case AMMO1 : return this->GameVars->PLAYER_AMMO[0];
        case AMMO2 : return this->GameVars->PLAYER_AMMO[1];
        case AMMO3 : return this->GameVars->PLAYER_AMMO[2];
        case AMMO4 : return this->GameVars->PLAYER_AMMO[3];
        case WEAPON1 : return this->GameVars->PLAYER_WEAPON[0];
        case WEAPON2 : return this->GameVars->PLAYER_WEAPON[1];
        case WEAPON3 : return this->GameVars->PLAYER_WEAPON[2];
        case WEAPON4 : return this->GameVars->PLAYER_WEAPON[3];
        case WEAPON5 : return this->GameVars->PLAYER_WEAPON[4];
        case WEAPON6 : return this->GameVars->PLAYER_WEAPON[5];
        case WEAPON7 : return this->GameVars->PLAYER_WEAPON[6];
        case KEY1 : return this->GameVars->PLAYER_KEY[0];
        case KEY2 : return this->GameVars->PLAYER_KEY[1];
        case KEY3 : return this->GameVars->PLAYER_KEY[2];
        default: return 0;
    }
}

int ViziaDoomController::getGameTic() { return this->GameVars->GAME_TIC; }
int ViziaDoomController::getMapTic() { return this->GameVars->MAP_TIC; }

int ViziaDoomController::getMapReward() { return this->GameVars->MAP_REWARD; }
int ViziaDoomController::getShapingReward() { return this->GameVars->SHAPING_REWARD; }

int ViziaDoomController::getMapKillCount() { return this->GameVars->MAP_KILLCOUNT; }
int ViziaDoomController::getMapItemCount() { return this->GameVars->MAP_ITEMCOUNT; }
int ViziaDoomController::getMapSecretCount() { return this->GameVars->MAP_SECRETCOUNT; }

bool ViziaDoomController::isPlayerDead() { return this->GameVars->PLAYER_DEAD; }

int ViziaDoomController::getPlayerKillCount() { return this->GameVars->PLAYER_KILLCOUNT; }
int ViziaDoomController::getPlayerItemCount() { return this->GameVars->PLAYER_ITEMCOUNT; }
int ViziaDoomController::getPlayerSecretCount() { return this->GameVars->PLAYER_SECRETCOUNT; }
int ViziaDoomController::getPlayerFragCount() { return this->GameVars->PLAYER_FRAGCOUNT; }

int ViziaDoomController::getPlayerHealth() { return this->GameVars->PLAYER_HEALTH; }
int ViziaDoomController::getPlayerArmor() { return this->GameVars->PLAYER_ARMOR; }

int ViziaDoomController::getPlayerAmmo1() { return this->GameVars->PLAYER_AMMO[0]; }
int ViziaDoomController::getPlayerAmmo2() { return this->GameVars->PLAYER_AMMO[1]; }
int ViziaDoomController::getPlayerAmmo3() { return this->GameVars->PLAYER_AMMO[2]; }
int ViziaDoomController::getPlayerAmmo4() { return this->GameVars->PLAYER_AMMO[3]; }

bool ViziaDoomController::getPlayerWeapon1() { return this->GameVars->PLAYER_WEAPON[0]; }
bool ViziaDoomController::getPlayerWeapon2() { return this->GameVars->PLAYER_WEAPON[1]; }
bool ViziaDoomController::getPlayerWeapon3() { return this->GameVars->PLAYER_WEAPON[2]; }
bool ViziaDoomController::getPlayerWeapon4() { return this->GameVars->PLAYER_WEAPON[3]; }
bool ViziaDoomController::getPlayerWeapon5() { return this->GameVars->PLAYER_WEAPON[4]; }
bool ViziaDoomController::getPlayerWeapon6() { return this->GameVars->PLAYER_WEAPON[5]; }
bool ViziaDoomController::getPlayerWeapon7() { return this->GameVars->PLAYER_WEAPON[6]; }

bool ViziaDoomController::getPlayerKey1() { return this->GameVars->PLAYER_KEY[0]; }
bool ViziaDoomController::getPlayerKey2() { return this->GameVars->PLAYER_KEY[1]; }
bool ViziaDoomController::getPlayerKey3() { return this->GameVars->PLAYER_KEY[2]; }

//PRIVATE

void ViziaDoomController::generateInstanceId(){
    std::string chars ="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";
    this->instanceId = "";

    srand(time(NULL));
    for(int i = 0; i < 10; ++i) {
        this->instanceId += chars[rand()%(chars.length()-1)];
    }
}

void ViziaDoomController::waitForDoomStart(){

    this->doomTic = true;

    MessageCommandStruct msg;

    unsigned int priority;
    bip::message_queue::size_type recvd_size;

    MQController->receive(&msg, sizeof(MessageCommandStruct), recvd_size, priority);

    switch(msg.code){
        case VIZIA_MSG_CODE_DOOM_READY :
        case VIZIA_MSG_CODE_DOOM_TIC :
            this->doomRunning = true;
            break;

        case VIZIA_MSG_CODE_DOOM_CLOSE :
            this->doomRunning = false;
            break;

        default : break;
    }

    this->doomTic = false;
}

void ViziaDoomController::waitForDoomTic(){
    if(doomRunning) {
        this->MQSend(VIZIA_MSG_CODE_TIC);

        this->doomTic = true;

        MessageCommandStruct msg;

        unsigned int priority;
        bip::message_queue::size_type recvd_size;

        bool nextTic = false;
        do {
            MQController->receive(&msg, sizeof(MessageCommandStruct), recvd_size, priority);
            switch (msg.code) {
                case VIZIA_MSG_CODE_DOOM_READY :
                case VIZIA_MSG_CODE_DOOM_TIC :
                    nextTic = true;
                    break;
                case VIZIA_MSG_CODE_DOOM_CLOSE :
                    this->close();
                    break;
                default :
                    break;
            }
        } while (!nextTic);

        this->doomTic = false;
    }
}

void ViziaDoomController::lunchDoom(){

    std::vector<std::string> args;

    //exe
    args.push_back(gamePath);

    //main wad
    args.push_back("-iwad");
    args.push_back(this->iwadPath);

    //skill
    args.push_back("-skill");
    args.push_back(b::lexical_cast<std::string>(this->skill));

    //wads
    if(this->filePath.length() != 0) {
        args.push_back("-file");
        args.push_back(this->filePath);
    }

    if(this->configPath.length() != 0) {
        args.push_back("-config");
        args.push_back(this->configPath);
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
    if(this->hud) args.push_back("10");
    else args.push_back("12");

    //weapon
    args.push_back("+r_drawplayersprites");
    if(this->weapon) args.push_back("1");
    else args.push_back("0");

    //crosshair
    args.push_back("+crosshair");
    if(this->crosshair){
        args.push_back("1");
        args.push_back("+crosshairhealth");
        args.push_back("0");
    }
    else args.push_back("0");

    //decals
    args.push_back("+cl_maxdecals");
    if(this->decals) args.push_back("128");
    else args.push_back("0");

    //particles
    args.push_back("+r_particles");
    if(this->decals) args.push_back("1");
    else args.push_back("0");

    //vizia args
    args.push_back("+vizia_controlled");
    args.push_back("1");

    args.push_back("+vizia_instance_id");
    args.push_back(this->instanceId);

    args.push_back("+vizia_singletic");
    args.push_back("1");

    args.push_back("+vizia_screen_format");
    args.push_back(b::lexical_cast<std::string>(this->screenFormat));

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
    args.push_back("true");

    args.push_back("+vid_vsync");
    args.push_back("false");

    //bpr::context ctx;
    //ctx.stdout_behavior = bpr::silence_stream();
    //this->doomProcess = bpr::execute(bpri::set_args(args));
    bpr::child doomProcess = bpr::execute(bpri::set_args(args), bpri::inherit_env());
}

//SM FUNCTIONS 
void ViziaDoomController::SMInit(){
    this->SMName = VIZIA_SM_NAME_BASE + instanceId;
    //bip::shared_memory_object::remove(this->SMName.c_str());

    this->SM = bip::shared_memory_object(bip::open_only, this->SMName.c_str(), bip::read_write);

    this->InputSMRegion = new bip::mapped_region(this->SM, bip::read_write, 0, sizeof(ViziaDoomController::InputStruct));
    this->Input = static_cast<ViziaDoomController::InputStruct *>(this->InputSMRegion->get_address());

    this->GameVarsSMRegion = new bip::mapped_region(this->SM, bip::read_only, sizeof(ViziaDoomController::InputStruct), sizeof(ViziaDoomController::GameVarsStruct));
    this->GameVars = static_cast<ViziaDoomController::GameVarsStruct *>(this->GameVarsSMRegion->get_address());

    this->screenWidth = this->GameVars->SCREEN_WIDTH;
    this->screenHeight = this->GameVars->SCREEN_HEIGHT;
    this->screenPitch = this->GameVars->SCREEN_PITCH;
    this->screenSize = this->GameVars->SCREEN_SIZE;
    this->screenFormat = (ViziaScreenFormat) this->GameVars->SCREEN_FORMAT;

    this->ScreenSMRegion = new bip::mapped_region(this->SM, bip::read_only, sizeof(ViziaDoomController::InputStruct) + sizeof(ViziaDoomController::GameVarsStruct), this->screenSize);
    this->Screen = static_cast<uint8_t *>(this->ScreenSMRegion->get_address());
}

void ViziaDoomController::SMClose(){
    delete(this->InputSMRegion);
    delete(this->GameVarsSMRegion);
    delete(this->ScreenSMRegion);
    bip::shared_memory_object::remove(this->SMName.c_str());
}

//MQ FUNCTIONS
void ViziaDoomController::MQInit(){

    this->MQControllerName = VIZIA_MQ_NAME_CTR_BASE + instanceId;
    this->MQDoomName = VIZIA_MQ_NAME_DOOM_BASE + instanceId;

    bip::message_queue::remove(this->MQControllerName.c_str());
    bip::message_queue::remove(this->MQDoomName.c_str());

    this->MQController = new bip::message_queue(bip::open_or_create, this->MQControllerName.c_str(), VIZIA_MQ_MAX_MSG_NUM, VIZIA_MQ_MAX_MSG_SIZE);
    this->MQDoom = new bip::message_queue(bip::open_or_create, this->MQDoomName.c_str(), VIZIA_MQ_MAX_MSG_NUM, VIZIA_MQ_MAX_MSG_SIZE);
}

void ViziaDoomController::MQSend(uint8_t code){
    MessageSignalStruct msg;
    msg.code = code;
    this->MQDoom->send(&msg, sizeof(MessageSignalStruct), 0);
}

bool ViziaDoomController::MQTrySend(uint8_t code){
    MessageSignalStruct msg;
    msg.code = code;
    return this->MQDoom->try_send(&msg, sizeof(MessageSignalStruct), 0);
}

void ViziaDoomController::MQSend(uint8_t code, const char * command){
    MessageCommandStruct msg;
    msg.code = code;
    strncpy(msg.command, command, VIZIA_MQ_MAX_CMD_LEN);
    this->MQDoom->send(&msg, sizeof(MessageCommandStruct), 0);
}

bool ViziaDoomController::MQTrySend(uint8_t code, const char * command){
    MessageCommandStruct msg;
    msg.code = code;
    strncpy(msg.command, command, VIZIA_MQ_MAX_CMD_LEN);
    return this->MQDoom->try_send(&msg, sizeof(MessageCommandStruct), 0);
}

void ViziaDoomController::MQRecv(void *msg, unsigned long &size, unsigned int &priority){
    this->MQController->receive(&msg, sizeof(MessageCommandStruct), size, priority);
}

bool ViziaDoomController::MQTryRecv(void *msg, unsigned long &size, unsigned int &priority){
    return this->MQController->try_receive(&msg, sizeof(MessageCommandStruct), size, priority);
}

void ViziaDoomController::MQClose(){
    //bip::message_queue::remove(this->MQDoomName.c_str());
    bip::message_queue::remove(this->MQControllerName.c_str());
}
