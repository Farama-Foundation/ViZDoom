#include "ViziaMain.h"

#include <boost/lexical_cast.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

float lastReward = 0.0f;//awful, temporary solution

unsigned int DoomTics2Ms (unsigned int tics){
    return (unsigned int)std::floor((float)1000/35 * tics);
}

unsigned int Ms2DoomTics (unsigned int ms){
    return (unsigned int)std::ceil((float)35/1000 * ms);
}

ViziaMain::ViziaMain(){
    initialized = false;
    /* Should usually be 0 but not always it seems. */
    this->state.vars = NULL;
    this->lastAction = NULL;
    this->doomController = new ViziaDoomController();
}

ViziaMain::~ViziaMain(){
    //TODO deleting stuff created in init
    this->close();
    delete this->doomController;
}

bool ViziaMain::loadConfig(std::string file){
    //TO DO
    return false;
}

bool ViziaMain::saveConfig(std::string file){
    //TO DO
    return false;
}

int ViziaMain::init(){
    if(initialized){
        std::cerr<<"Initialization has already been done. Aborting init.";
        return -1;
    }
    else{
        initialized = true;
    }

    if( this->stateAvailableVars.size()){
        this->state.vars = new int[this->stateAvailableVars.size()];
    }
   
    /* set all if none are set */
    this->lastAction = new bool[this->availableButtons.size()];

    this->doomController->init();

    /* Initialize state format */
    int y = this->doomController->getScreenWidth();
    int x = this->doomController->getScreenHeight();
    int channels = 3;
    int varLen = this->stateAvailableVars.size();
    this->stateFormat = StateFormat(channels, x, y, varLen);

    /* TODO fill state here? */    
    return 0;
}

void ViziaMain::close(){
    this->doomController->close();

    delete[](this->state.vars);
    this->state.vars= NULL;
    delete[](this->lastAction);
    this->lastAction = NULL;
}

void ViziaMain::newEpisode(){
    this->doomController->restartMap();
}

float ViziaMain::makeAction(std::vector<bool>& actions){

    int j = 0;
    for (std::vector<ViziaButton>::iterator i = this->availableButtons.begin() ; i != this->availableButtons.end(); ++i, ++j){
        this->lastAction[j] = actions[j];
        this->doomController->setButtonState(*i, actions[j]);
    }

    this->doomController->tic();

    /* Updates vars */
    j = 0;
    for (std::vector<ViziaGameVar>::iterator i = this->stateAvailableVars.begin() ; i != this->stateAvailableVars.end(); ++i, ++j){
        this->state.vars[j] = this->doomController->getGameVar(*i);
    }

    /* Update float rgb image */
    this->state.number = this->doomController->getMapTic();
    this->state.imageBuffer = this->doomController->getScreen();
    this->state.imageWidth = this->doomController->getScreenWidth();
    this->state.imageHeight = this->doomController->getScreenHeight();
    this->state.imagePitch = this->doomController->getScreenPitch();
    
    /* Return tic reward */    
    
    float mapReward = (float) this->doomController->getMapReward();
    float reward = mapReward - lastReward;
    lastReward = mapReward;
      
    return reward;
}

ViziaMain::State ViziaMain::getState(){
    return this->state;
}

bool * ViziaMain::getlastAction(){ return this->lastAction; }

bool ViziaMain::isNewEpisode(){
    return this->doomController->isMapFirstTic();
}

bool ViziaMain::isEpisodeFinished(){
    return this->doomController->isMapLastTic() || this->doomController->isPlayerDead();
}

void ViziaMain::addAvailableButton(ViziaButton button){
    if(std::find(this->availableButtons.begin(),this->availableButtons.end(), button) == this->availableButtons.end()) {
        this->availableButtons.push_back(button);
    }
}

//void ViziaMain::addAvailableButton(std::string button){
//    this->addAvailableButton(ViziaDoomController::getButtonId(button));
//}

void ViziaMain::addStateAvailableVar(ViziaGameVar var){
    if(std::find(this->stateAvailableVars.begin(),this->stateAvailableVars.end(), var) == this->stateAvailableVars.end()) {
        this->stateAvailableVars.push_back(var);
    }
}

//void ViziaMain::addStateAvailableVar(std::string var){
//    this->addStateAvailableVar(ViziaDoomController::getGameVarId(var));
//}

const ViziaDoomController* ViziaMain::getController(){ return this->doomController; }

void ViziaMain::setDoomGamePath(std::string path){ this->doomController->setGamePath(path); }
void ViziaMain::setDoomIwadPath(std::string path){ this->doomController->setIwadPath(path); }
void ViziaMain::setDoomFilePath(std::string path){ this->doomController->setFilePath(path); }
void ViziaMain::setDoomMap(std::string map){ this->doomController->setMap(map); }
void ViziaMain::setDoomSkill(int skill){ this->doomController->setSkill(skill); }
void ViziaMain::setDoomConfigPath(std::string path){ this->doomController->setConfigPath(path); }

void ViziaMain::setAutoNewEpisode(bool set){ this->doomController->setAutoMapRestart(set); }
void ViziaMain::setNewEpisodeOnTimeout(bool set){ this->doomController->setAutoMapRestartOnTimeout(set); }
void ViziaMain::setNewEpisodeOnPlayerDeath(bool set){ this->doomController->setAutoMapRestartOnTimeout(set); }

void ViziaMain::setEpisodeTimeoutInMiliseconds(unsigned int ms){
    this->doomController->setMapTimeout(Ms2DoomTics(ms));
}

void ViziaMain::setEpisodeTimeoutInDoomTics(unsigned int tics){
    this->doomController->setMapTimeout(tics);
}

void ViziaMain::setScreenResolution(unsigned int width, unsigned int height){ this->doomController->setScreenResolution(width, height); }
void ViziaMain::setScreenWidth(unsigned int width){ this->doomController->setScreenWidth(width); }
void ViziaMain::setScreenHeight(unsigned int height){ this->doomController->setScreenHeight(height); }
void ViziaMain::setScreenFormat(ViziaScreenFormat format){ this->doomController->setScreenFormat(format); }
void ViziaMain::setRenderHud(bool hud){ this->doomController->setRenderHud(hud); }
void ViziaMain::setRenderWeapon(bool weapon){ this->doomController->setRenderWeapon(weapon); }
void ViziaMain::setRenderCrosshair(bool crosshair){ this->doomController->setRenderCrosshair(crosshair); }
void ViziaMain::setRenderDecals(bool decals){ this->doomController->setRenderDecals(decals); }
void ViziaMain::setRenderParticles(bool particles){ this->doomController->setRenderParticles(particles); }

int ViziaMain::getScreenWidth(){ return this->doomController->getScreenWidth(); }
int ViziaMain::getScreenHeight(){ return this->doomController->getScreenHeight(); }
size_t ViziaMain::getScreenPitch(){ return this->doomController->getScreenPitch(); }
size_t ViziaMain::getScreenSize(){ return this->doomController->getScreenSize(); }
ViziaScreenFormat ViziaMain::getScreenFormat(){ return this->doomController->getScreenFormat(); }

ViziaMain::StateFormat ViziaMain::getStateFormat()
{
    return this->stateFormat;
}
int ViziaMain::getActionFormat()
{
    return this->availableButtons.size();
}
