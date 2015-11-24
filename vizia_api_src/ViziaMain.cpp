#include "ViziaMain.h"

#include <boost/lexical_cast.hpp>
#include <cmath>
#include <iostream>
#include <vector>

unsigned int DoomTics2Ms (unsigned int tics){
    return (unsigned int)std::floor((float)1000/35 * tics);
}

unsigned int Ms2DoomTics (unsigned int ms){
    return (unsigned int)std::ceil((float)35/1000 * ms);
}

ViziaMain::ViziaMain(){
    initialized = false;
    rgbConversion = false;
    this->doomController = new ViziaDoomController();
}

ViziaMain::~ViziaMain(){
    //TODO deleting stuff created in init
    this->close();
    delete(this->doomController);
}

bool ViziaMain::loadConfig(std::string file){
    //TO DO
    return false;
}

bool ViziaMain::saveConfig(std::string file){
    //TO DO
    return false;
}

void ViziaMain::init(){
    if(initialized){
        std::cerr<<"Initialization Has already been done. Aborting init.";
    }
    else{
        initialized = true;
    }
    this->stateVars = new int[this->stateAvailableVars.size()];
    this->lastActions = new bool[this->availableActions.size()];

    this->doomController->init();

    int j = 0;
    for (std::vector<int>::iterator i = this->stateAvailableVars.begin() ; i != this->stateAvailableVars.end(); ++i, ++j){
        this->stateVars[j] = this->doomController->getGameVar(*i);
    }
}

void ViziaMain::close(){
    this->doomController->close();
    delete(this->stateVars);
    delete(this->lastActions);
}

void ViziaMain::newEpisode(){
    this->doomController->restartMap();
}

float ViziaMain::makeAction(std::vector<bool>& actions){

    int j = 0;
    for (std::vector<int>::iterator i = this->availableActions.begin() ; i != this->availableActions.end(); ++i, ++j){
        this->lastActions[j] = actions[j];
        this->doomController->setButtonState(*i, actions[j]);
    }

    this->doomController->tic();

    j = 0;
    for (std::vector<int>::iterator i = this->stateAvailableVars.begin() ; i != this->stateAvailableVars.end(); ++i, ++j){
        this->stateVars[j] = this->doomController->getGameVar(*i);
    }
    //TODO return reward
    return 0.0;
}

ViziaMain::State ViziaMain::getState(){
    ViziaMain::State state;
    state.number = this->doomController->getMapTic();
    state.vars = this->stateVars;

    state.imageBuffer = this->doomController->getScreen();
    if ( this->rgbConversion ){
        //TODO
    }
    return state;
}

bool * ViziaMain::getLastActions(){ return this->lastActions; }

bool ViziaMain::isNewEpisode(){
    return this->doomController->isMapFirstTic();
}

bool ViziaMain::isEpisodeFinished(){
    return this->doomController->isMapLastTic() || this->doomController->isPlayerDead();
}

void ViziaMain::addAvailableAction(int action){
    this->availableActions.push_back(action);
}

void ViziaMain::addAvailableAction(std::string action){
    this->availableActions.push_back(ViziaDoomController::getButtonId(action));
}

void ViziaMain::addStateAvailableVar(int var){
    this->stateAvailableVars.push_back(var);
}

void ViziaMain::addStateAvailableVar(std::string var){
    this->stateAvailableVars.push_back(ViziaDoomController::getGameVarId(var));
}

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

void ViziaMain::setScreenResolution(int width, int height){ this->doomController->setScreenResolution(width, height); }
void ViziaMain::setScreenWidth(int width){ this->doomController->setScreenWidth(width); }
void ViziaMain::setScreenHeight(int height){ this->doomController->setScreenHeight(height); }
void ViziaMain::setScreenFormat(int format){ this->doomController->setScreenFormat(format); }
void ViziaMain::setRenderHud(bool hud){ this->doomController->setRenderHud(hud); }
void ViziaMain::setRenderWeapon(bool weapon){ this->doomController->setRenderWeapon(weapon); }
void ViziaMain::setRenderCrosshair(bool crosshair){ this->doomController->setRenderCrosshair(crosshair); }
void ViziaMain::setRenderDecals(bool decals){ this->doomController->setRenderDecals(decals); }
void ViziaMain::setRenderParticles(bool particles){ this->doomController->setRenderParticles(particles); }
void ViziaMain::setRGBConversion(bool rgbOn){ this->rgbConversion = rgbOn;}

int ViziaMain::getScreenWidth(){ return this->doomController->getScreenWidth(); }
int ViziaMain::getScreenHeight(){ return this->doomController->getScreenHeight(); }
int ViziaMain::getScreenPitch(){ return this->doomController->getScreenPitch(); }
int ViziaMain::getScreenSize(){ return this->doomController->getScreenSize(); }
int ViziaMain::getScreenFormat(){ return this->doomController->getScreenFormat(); }
