#include "ViziaDoomGame.h"

#include <boost/lexical_cast.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

namespace Vizia {

    unsigned int DoomTics2Ms(unsigned int tics) {
        return (unsigned int) std::floor((float) 1000 / 35 * tics);
    }

    unsigned int Ms2DoomTics(unsigned int ms) {
        return (unsigned int) std::ceil((float) 35 / 1000 * ms);
    }

    DoomGame::DoomGame() {
        this->running = false;

        this->lastReward = 0;

        this->lastMapReward = 0;
        this->lastShapingReward = 0;

        this->deathPenalty = 0;
        this->livingReward = 0;
        this->summaryReward = 0;

        this->doomController = new DoomController();
    }

    DoomGame::~DoomGame() {
        this->close();
        delete this->doomController;
    }

    bool DoomGame::loadConfig(std::string file) {
        //TO DO
        return false;
    }

    bool DoomGame::saveConfig(std::string file) {
        //TO DO
        return false;
    }

    bool DoomGame::init() {
        if (!this->running) {

            this->state.vars.resize(this->stateAvailableVars.size());
            /* set all if none are set */
            this->lastAction.resize(this->availableButtons.size());

            try {
                this->running = this->doomController->init();
            }
            catch(const Exception &e){ throw; }

            /* Initialize state format */
            int y = this->doomController->getScreenWidth();
            int x = this->doomController->getScreenHeight();
            int channels = 3;
            int varLen = this->stateAvailableVars.size();
            this->stateFormat = StateFormat(channels, x, y, varLen);

            return running;
        }
        else return false;
    }

    void DoomGame::close() {
        this->doomController->close();
        this->state.vars.clear();
        this->lastAction.clear();

        this->running = false;
    }

    bool DoomGame::isRunning(){
        return this->running && this->doomController->isDoomRunning();
    }

    void DoomGame::newEpisode() {

        if(!this->isRunning()) throw DoomIsNotRunningException();

        this->doomController->restartMap();

        this->lastReward = 0;
        this->lastMapReward = 0;
        this->lastShapingReward = 0;
        this->summaryReward = 0;
    }

    float DoomGame::makeAction(std::vector<bool> &actions) {

        if(!this->isRunning()) throw DoomIsNotRunningException();

        int j = 0;
        try {
            for (std::vector<Button>::iterator i = this->availableButtons.begin();
                 i != this->availableButtons.end(); ++i, ++j) {
                this->lastAction[j] = actions[j];
                this->doomController->setButtonState(*i, actions[j]);
            }
        }
        catch (...){ throw SharedMemoryException(); }

        try {
            this->doomController->tic();
        }
        catch(const Exception &e){ throw; }

        int reward = 0;

        try {
            /* Updates vars */
            j = 0;
            for (std::vector<GameVar>::iterator i = this->stateAvailableVars.begin();
                 i != this->stateAvailableVars.end(); ++i, ++j) {
                this->state.vars[j] = this->doomController->getGameVar(*i);
            }

            /* Update float rgb image */
            this->state.number = this->doomController->getMapTic();
            this->state.imageBuffer = this->doomController->getScreen();
            this->state.imageWidth = this->doomController->getScreenWidth();
            this->state.imageHeight = this->doomController->getScreenHeight();
            this->state.imagePitch = this->doomController->getScreenPitch();

            /* Return tic reward */

            int mapReward = this->doomController->getMapReward();
            int shapingReward = this->doomController->getMapShapingReward();

            int reward = (mapReward - this->lastMapReward) + this->livingReward;
            if(this->includeShapingReward) reward += (shapingReward - this->lastShapingReward);
            if(this->doomController->isPlayerDead()) reward -= this->deathPenalty;

            this->lastMapReward = mapReward;
            this->lastShapingReward = shapingReward;

            this->summaryReward += reward;

            this->lastReward = reward;
        }
        catch (...){ throw SharedMemoryException(); }

        return reward;
    }

    DoomGame::State DoomGame::getState() {
        return this->state;
    }

    std::vector<bool> DoomGame::getLastAction() { return this->lastAction; }

    bool DoomGame::isNewEpisode() {
        if(!this->isRunning()) throw DoomIsNotRunningException();

        return this->doomController->isMapFirstTic();
    }

    bool DoomGame::isEpisodeFinished() {
        if(!this->isRunning()) throw DoomIsNotRunningException();

        return this->doomController->isMapLastTic()
               || this->doomController->isPlayerDead()
               || this->doomController->isMapEnded();
    }

    void DoomGame::addAvailableButton(Button button) {
        if (std::find(this->availableButtons.begin(), this->availableButtons.end(), button) ==
            this->availableButtons.end()) {
            this->availableButtons.push_back(button);
        }
    }

//void DoomGame::addAvailableButton(std::string button){
//    this->addAvailableButton(ViziaDoomController::getButtonId(button));
//}

    void DoomGame::addStateAvailableVar(GameVar var) {
        if (std::find(this->stateAvailableVars.begin(), this->stateAvailableVars.end(), var) ==
            this->stateAvailableVars.end()) {
            this->stateAvailableVars.push_back(var);
        }
    }

//void DoomGame::addStateAvailableVar(std::string var){
//    this->addStateAvailableVar(ViziaDoomController::getGameVarId(var));
//}

    const DoomController* DoomGame::getController() { return this->doomController; }

    int DoomGame::getGameVar(GameVar var){
        if(!this->isRunning()) throw DoomIsNotRunningException();

        return this->doomController->getGameVar(var);
    }

    void DoomGame::setDoomGamePath(std::string path) { this->doomController->setGamePath(path); }
    void DoomGame::setDoomIwadPath(std::string path) { this->doomController->setIwadPath(path); }
    void DoomGame::setDoomFilePath(std::string path) { this->doomController->setFilePath(path); }
    void DoomGame::setDoomMap(std::string map) { this->doomController->setMap(map); }
    void DoomGame::setDoomSkill(int skill) { this->doomController->setSkill(skill); }
    void DoomGame::setDoomConfigPath(std::string path) { this->doomController->setConfigPath(path); }

    void DoomGame::setAutoNewEpisode(bool set) { this->doomController->setAutoMapRestart(set); }
    void DoomGame::setNewEpisodeOnTimeout(bool set) { this->doomController->setAutoMapRestartOnTimeout(set); }
    void DoomGame::setNewEpisodeOnPlayerDeath(bool set) { this->doomController->setAutoMapRestartOnTimeout(set); }
    void DoomGame::setNewEpisodeOnMapEnd(bool set) { this->doomController->setAutoMapRestartOnMapEnd(set); }

//void DoomGame::setEpisodeStartTimeInMiliseconds(unsigned int ms){
//    this->doomController->setMapStartTime(Ms2DoomTics(ms));
//}
//
//void DoomGame::setEpisodeStartTimeInDoomTics(unsigned int tics){
//    this->doomController->setMapStartTime(tics);
//}

    unsigned int DoomGame::getEpisodeTimeoutInMiliseconds(){ return DoomTics2Ms(this->doomController->getMapTimeout()); }
    void DoomGame::setEpisodeTimeoutInMiliseconds(unsigned int ms) {
        this->doomController->setMapTimeout(Ms2DoomTics(ms));
    }

    unsigned int DoomGame::getEpisodeTimeoutInDoomTics(){ return this->doomController->getMapTimeout(); }
    void DoomGame::setEpisodeTimeoutInDoomTics(unsigned int tics) {
        this->doomController->setMapTimeout(tics);
    }

    bool DoomGame::isShapingRewardIncluded() { return this->includeShapingReward; }
    void DoomGame::setShapingRewardIncluded(bool include){ this->includeShapingReward = include; };

    int DoomGame::getLivingReward() { return this->livingReward; }
    void DoomGame::setLivingReward(int livingReward) { this->livingReward = livingReward; }

    int DoomGame::getDeathPenalty() { return this->deathPenalty; }
    void DoomGame::setDeathPenalty(int deathPenalty) { this->deathPenalty = deathPenalty; }

    int DoomGame::getLastReward(){ return this->lastReward; }
    int DoomGame::getSummaryReward() { return this->summaryReward; }

    void DoomGame::setScreenResolution(unsigned int width, unsigned int height) {
        this->doomController->setScreenResolution(width, height);
    }

    void DoomGame::setScreenWidth(unsigned int width) { this->doomController->setScreenWidth(width); }
    void DoomGame::setScreenHeight(unsigned int height) { this->doomController->setScreenHeight(height); }
    void DoomGame::setScreenFormat(ScreenFormat format) { this->doomController->setScreenFormat(format); }
    void DoomGame::setRenderHud(bool hud) { this->doomController->setRenderHud(hud); }
    void DoomGame::setRenderWeapon(bool weapon) { this->doomController->setRenderWeapon(weapon); }
    void DoomGame::setRenderCrosshair(bool crosshair) { this->doomController->setRenderCrosshair(crosshair); }
    void DoomGame::setRenderDecals(bool decals) { this->doomController->setRenderDecals(decals); }
    void DoomGame::setRenderParticles(bool particles) { this->doomController->setRenderParticles(particles); }
    int DoomGame::getScreenWidth() { return this->doomController->getScreenWidth(); }
    int DoomGame::getScreenHeight() { return this->doomController->getScreenHeight(); }
    size_t DoomGame::getScreenPitch() { return this->doomController->getScreenPitch(); }
    size_t DoomGame::getScreenSize() { return this->doomController->getScreenSize(); }
    ScreenFormat DoomGame::getScreenFormat() { return this->doomController->getScreenFormat(); }

    DoomGame::StateFormat DoomGame::getStateFormat() {
        return this->stateFormat;
    }

    int DoomGame::getActionFormat() {
        return this->availableButtons.size();
    }
}
