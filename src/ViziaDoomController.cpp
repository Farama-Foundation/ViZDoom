#include "ViziaDoomController.h"

#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>

//PUBLIC FUNCTIONS

ViziaDoomController::ViziaDoomController(){
    this->screenWidth = 640;
    this->screenHeight = 480;
    this->screenSize = screenWidth * screenHeight;

    this->gamePath = "./zdoom";
    this->iwadPath = "doom2.wad";
    this->file = "";
    this->map = "map01";
    this->skill = 1;

    this->hud = true;
    this->weapon = true;
    this->crosshair = false;
    this->decals = true;
    this->particles = true;

    this->autoRestartOnTimeout = true;
    this->autoRestartOnPlayersDeath = true;
    this->mapTimeout = 0;
    this->mapRestartCount = 0;

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
        this->SMInit();

        //this->lunchDoom();
        doomThread = new b::thread(b::bind(&ViziaDoomController::lunchDoom, this));
        this->waitForDoom();
    }

    return true;
}

bool ViziaDoomController::close(){

//    if(this->doomRunning) {
//        bpr::terminate(this->doomProcess);
//    }

    this->MQSend(VIZIA_MSG_CODE_CLOSE);

    doomThread->interrupt();
    doomThread->join();

    this->SMClose();
    this->MQClose();

    this->doomRunning = false;

    return true;
}

bool ViziaDoomController::tic(){

    if(doomRunning) {

        if(this->autoRestartOnPlayersDeath && this->GameVars->PLAYER_DEAD){
            this->restartMap();
        }
        else if(this->autoRestartOnTimeout && this->mapTimeout > 0 && this->GameVars->MAP_TIC >= mapTimeout-1){
            this->restartMap();
        }
        else{
            this->mapRestarting = false;
        }

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

    return true;
}

bool ViziaDoomController::update(){
    return this->tic();
}

void ViziaDoomController::restartMap(){
    if(this->doomRunning && !this->mapRestarting) {
        this->sendCommand("map " + this->map);
        ++this->mapRestartCount;
        this->mapRestarting = true;
    }
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
void ViziaDoomController::setFilePath(std::string path){ this->file = path; }

void ViziaDoomController::setMap(std::string map){
    if(map != this->map) this->mapRestartCount = 0;
    this->map = map;
    if(this->doomRunning){
        this->sendCommand("map "+this->map);
    }
}

void ViziaDoomController::setSkill(int skill){
    this->skill = skill;
    if(this->doomRunning){
        this->sendCommand("skill "+this->skill);
        //this->resetMap();
    }
}

void ViziaDoomController::setAutoMapRestartOnTimeout(bool set){
    this->autoRestartOnTimeout = set;
}

void ViziaDoomController::setAutoMapRestartOnPlayerDeath(bool set){
    this->autoRestartOnPlayersDeath = set;
}

void ViziaDoomController::setMapTimeout(unsigned int tics){
    this->mapTimeout = tics;
}

void ViziaDoomController::setScreenSize(int screenWidth, int screenHeight){
    this->screenWidth = screenWidth;
    this->screenHeight = screenHeight;
    this->screenSize = screenWidth*screenHeight;

//    if(this->doomRunning){
//        this->sendCommand("vid_defwidth "+b::lexical_cast<std::string>(this->screenWidth));
//        this->sendCommand("vid_defheight "+b::lexical_cast<std::string>(this->screenHeight));
//    }
}

void ViziaDoomController::showHud(bool hud){
    this->hud = hud;
    if(this->doomRunning){
        if(this->hud) this->sendCommand("screenblocks 10");
        else this->sendCommand("screenblocks 12");
    }
}

void ViziaDoomController::showWeapon(bool weapon){
    this->weapon = weapon;
    if(this->doomRunning){
        if(this->weapon) this->sendCommand("r_drawplayersprites 1");
        else this->sendCommand("r_drawplayersprites 1");
    }
}

void ViziaDoomController::showCrosshair(bool crosshair){
    this->crosshair = crosshair;
    if(this->doomRunning){
        if(this->crosshair){
            this->sendCommand("crosshairhealth false");
            this->sendCommand("crosshair 1");
        }
        else this->sendCommand("crosshair 0");
    }
}

void ViziaDoomController::showDecals(bool decals){
    this->decals = decals;
    if(this->doomRunning){
        if(this->decals) this->sendCommand("cl_maxdecals 128");
        else this->sendCommand("cl_maxdecals 0");
    }
}

void ViziaDoomController::showParticles(bool particles){
    this->particles = particles;
    if(this->doomRunning){
        if(this->particles) this->sendCommand("r_particles 1");
        else this->sendCommand("r_particles 0");
    }
}


//PRIVATE

void ViziaDoomController::waitForDoom(){

    this->doomTic = true;

    MessageCommandStruct msg;

    unsigned int priority;
    bip::message_queue::size_type recvd_size;

    bool nextTic = false;

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
    if(this->file.length() != 0) {
        args.push_back("-file");
        args.push_back(this->file);
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

    //TEMP CONST ARGS

    args.push_back("+vizia_controlled");
    args.push_back("1");

    args.push_back("+vizia_instance_id");
    args.push_back(this->instanceId);

    args.push_back("+wipetype");
    args.push_back("0");

    args.push_back("-noidle");
    args.push_back("-nojoy");
    args.push_back("-nosound");

    args.push_back("+cl_capfps");
    args.push_back("true");

    args.push_back("+vid_vsync");
    args.push_back("false");

    //bpr::context ctx;
    //ctx.stdout_behavior = bpr::silence_stream();
    //this->doomProcess = bpr::execute(bpri::set_args(args));
    bpr::child doomProcess = bpr::execute(bpri::set_args(args), bpri::inherit_env());
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

void ViziaDoomController::setButtonState(int button, bool state){
    if( button < V_BT_SIZE ) this->Input->BT[button] = state;
}

void ViziaDoomController::setKeyState(int key, bool state){
    if( key < V_BT_SIZE ) this->Input->BT[key] = state;
}

void ViziaDoomController::toggleButtonState(int button){
    if( button < V_BT_SIZE ) this->Input->BT[button] = !this->Input->BT[button];
}

void ViziaDoomController::toggleKeyState(int key){
    if( key < V_BT_SIZE ) this->Input->BT[key] = !this->Input->BT[key];
}

int ViziaDoomController::getGameTic() { return this->GameVars->GAME_TIC; }
int ViziaDoomController::getMapTic() { return this->GameVars->MAP_TIC; }

int ViziaDoomController::getMapKillCount() { return this->GameVars->MAP_KILLCOUNT; }
int ViziaDoomController::getMapItemCount() { return this->GameVars->MAP_ITEMCOUNT; }
int ViziaDoomController::getMapSecretCount() { return this->GameVars->MAP_SECRETCOUNT; }

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

//SM FUNCTIONS 
void ViziaDoomController::SMInit(){
    this->SMName = VIZIA_SM_NAME_BASE + instanceId;
    bip::shared_memory_object::remove(this->SMName.c_str());

    this->SM = bip::shared_memory_object(bip::open_or_create, this->SMName.c_str(), bip::read_write);
    this->SMSetSize(screenWidth, screenHeight);

    this->InputSMRegion = new bip::mapped_region(this->SM, bip::read_write, this->SMGetInputRegionBeginning(), sizeof(ViziaDoomController::InputStruct));
    this->Input = static_cast<ViziaDoomController::InputStruct *>(this->InputSMRegion->get_address());

    this->GameVarsSMRegion = new bip::mapped_region(this->SM, bip::read_only, this->SMGetGameVarsRegionBeginning(), sizeof(ViziaDoomController::GameVarsStruct));
    this->GameVars = static_cast<ViziaDoomController::GameVarsStruct *>(this->GameVarsSMRegion->get_address());

    this->ScreenSMRegion = new bip::mapped_region(this->SM, bip::read_only, this->SMGetScreenRegionBeginning(), sizeof(uint8_t) * this->screenSize);
    this->Screen = static_cast<uint8_t *>(this->ScreenSMRegion->get_address());
}

void ViziaDoomController::SMSetSize(int screenWidth, int screenHeight){
    this->SMSize = sizeof(InputStruct) + sizeof(GameVarsStruct) + (sizeof(uint8_t) * screenWidth * screenHeight);
    this->SM.truncate(this->SMSize);
}

size_t ViziaDoomController::SMGetInputRegionBeginning(){
    return 0;
}

size_t ViziaDoomController::SMGetGameVarsRegionBeginning(){
    return sizeof(InputStruct);
}

size_t ViziaDoomController::SMGetScreenRegionBeginning(){
    return sizeof(InputStruct) + sizeof(GameVarsStruct);
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
    //bip::message_queue::remove(VIZIA_MQ_NAME_CTR);
    bip::message_queue::remove(this->MQControllerName.c_str());
}
