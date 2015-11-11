#include "ViziaDoomController.h"

//PUBLIC FUNCTIONS

ViziaDoomController::ViziaDoomController(){
    this->screenWidth = 640;
    this->screenHeight = 480;
    this->screenSize = screenWidth * screenHeight;

    this->gamePath = "./zdoom";
    this->iwadPath = "doom2.wad";
    this->file = "";
    this->map = "MAP01";
    this->skill = 1;
}

ViziaDoomController::ViziaDoomController(std::string iwad, std::string file, std::string map, int screenWidth, int screenHeight){
    this->screenWidth = screenWidth;
    this->screenHeight = screenHeight;
    this->screenSize = screenWidth * screenHeight;

    this->gamePath = "./zdoom";
    this->iwadPath = iwad;
    this->file = file;
    this->map = map;
    this->skill = 1;
}

ViziaDoomController::~ViziaDoomController(){
    this->close();
}

bool ViziaDoomController::init(){

    if(this->iwadPath.length() != 0 && this->map.length() != 0) {
        this->MQInit();
        this->SMInit();

        this->lunchDoom();
        this->waitForDoom();
    }

    return true;
}

bool ViziaDoomController::close(){
    this->SMClose();
    this->MQClose();

    return true;
}

void ViziaDoomController::setScreenSize(int screenWidth, int screenHeight){
    this->screenWidth = screenWidth;
    this->screenHeight = screenHeight;
    this->screenSize = screenWidth*screenHeight;
}

void ViziaDoomController::setGamePath(std::string path){ this->gamePath = path; }
void ViziaDoomController::setIwadPath(std::string path){ this->iwadPath = path; }
void ViziaDoomController::setFilePath(std::string path){ this->file = path; }

void ViziaDoomController::setMap(std::string map){
    this->map = map;
    if(this->doomRunning){
        this->sendCommand("MAP "+this->map);
    }
}

void ViziaDoomController::setSkill(int skill){
    this->skill = skill;
    if(this->doomRunning){
        this->sendCommand("SKILL "+this->skill);
        this->resetMap();
    }
}

void ViziaDoomController::setFrameRate() {} //TO DO
void ViziaDoomController::setHUD() {} //TO DO
void ViziaDoomController::setCrosshair() {} //TO DO

bool ViziaDoomController::tic(){
    MessageCommandStruct msg;

    unsigned int priority;
    bip::message_queue::size_type recvd_size;

    bool nextTic = false;
    do {
        MQ->receive(&msg, sizeof(MessageCommandStruct), recvd_size, priority);
        switch(msg.code){
            case VIZIA_MSG_CODE_DOOM_READY :
            case VIZIA_MSG_CODE_DOOM_TIC :
                nextTic = true;
                break;
            default : break;
        }
    }while(!nextTic);

    return true;
}

bool ViziaDoomController::update(){
    return tic();
}

void ViziaDoomController::resetMap(){
    this->sendCommand("MAP "+this->map);
}

void ViziaDoomController::restartGame(){
    //TO DO
}

void ViziaDoomController::sendCommand(std::string command){
    this->MQSend(VIZIA_MSG_CODE_COMMAND, command.c_str());
}

//PRIVATE

void ViziaDoomController::waitForDoom(){
    this->tic();

    doomRunning = true;
}

void ViziaDoomController::lunchDoom(){
//
//    execute(set_args(args));
//
//    std::vector<std::string> args;
//    args.push_back(gamePath);
//    args.push_back("-iwad");
//    args.push_back(this->iwadPath);
//    args.push_back("-skill");
//    //args.push_back(this->skill);
//    args.push_back("1");
//    if(this->file.length() != 0) {
//        args.push_back("-file");
//        args.push_back(this->file);
//    }
//    args.push_back("+map");
//    args.push_back(this->map);
//
//    //bpr::context ctx;
//    //ctx.stdout_behavior = bpr::silence_stream();
//
//    doomProcess = bpr::execute(set_args(args));
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

int ViziaDoomController::getGameTic() { return this->GameVars->TIC; }

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

//SM FUNCTIONS 
void ViziaDoomController::SMInit(){
    bip::shared_memory_object::remove(VIZIA_SM_NAME);

    this->SM = new bip::shared_memory_object(bip::open_or_create, VIZIA_SM_NAME, bip::read_write);
    this->SMSetSize(screenWidth, screenHeight);

    this->InputSMRegion = new bip::mapped_region(this->SM, bip::read_write, this->SMGetInputRegionBeginning(), sizeof(ViziaDoomController::InputStruct));
    this->Input = static_cast<ViziaDoomController::InputStruct *>(this->InputSMRegion->get_address());

    this->GameVarsSMRegion = new bip::mapped_region(this->SM, bip::read_write, this->SMGetGameVarsRegionBeginning(), sizeof(ViziaDoomController::GameVarsStruct));
    this->GameVars = static_cast<ViziaDoomController::GameVarsStruct *>(this->GameVarsSMRegion->get_address());

    this->ScreenSMRegion = new bip::mapped_region(this->SM, bip::read_write, this->SMGetScreenRegionBeginning(), sizeof(uint8_t) * this->screenSize);
    this->Screen = static_cast<uint8_t *>(this->ScreenSMRegion->get_address());

}

void ViziaDoomController::SMSetSize(int screenWidth, int screenHeight){
    this->SMSize = sizeof(InputStruct) + sizeof(GameVarsStruct) + (sizeof(uint8_t) * screenWidth * screenHeight);
    this->SM->truncate(this->SMSize);
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
    bip::shared_memory_object::remove(VIZIA_SM_NAME);
}

//MQ FUNCTIONS
void ViziaDoomController::MQInit(){
    bip::message_queue::remove(VIZIA_MQ_NAME);
    this->MQ = new bip::message_queue(bip::open_or_create, VIZIA_MQ_NAME, VIZIA_MQ_MAX_MSG_NUM, VIZIA_MQ_MAX_MSG_SIZE);
}

void ViziaDoomController::MQSend(uint8_t code){
    MessageSignalStruct msg;
    msg.code = code;
    this->MQ->send(&msg, sizeof(MessageSignalStruct), 0);
}

bool ViziaDoomController::MQTrySend(uint8_t code){
    MessageSignalStruct msg;
    msg.code = code;
    return this->MQ->try_send(&msg, sizeof(MessageSignalStruct), 0);
}

void ViziaDoomController::MQSend(uint8_t code, const char * command){
    MessageCommandStruct msg;
    msg.code = code;
    strncpy(msg.command, command, VIZIA_MQ_MAX_CMD_LEN);
    this->MQ->send(&msg, sizeof(MessageCommandStruct), 0);
}

bool ViziaDoomController::MQTrySend(uint8_t code, const char * command){
    MessageCommandStruct msg;
    msg.code = code;
    strncpy(msg.command, command, VIZIA_MQ_MAX_CMD_LEN);
    return this->MQ->try_send(&msg, sizeof(MessageCommandStruct), 0);
}

void ViziaDoomController::MQRecv(void *msg, unsigned long &size, unsigned int &priority){
    this->MQ->receive(&msg, sizeof(MessageCommandStruct), size, priority);
}

bool ViziaDoomController::MQTryRecv(void *msg, unsigned long &size, unsigned int &priority){
    return this->MQ->try_receive(&msg, sizeof(MessageCommandStruct), size, priority);
}

void ViziaDoomController::MQClose(){
    bip::message_queue::remove(VIZIA_MQ_NAME);
}
