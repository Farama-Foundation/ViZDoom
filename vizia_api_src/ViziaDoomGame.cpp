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

    float DoomFixedToFloat(int doomFixed)
    {
        float res = float(doomFixed)/65536.0;
        return res;
    }

    DoomGame::DoomGame() {
        this->running = false;
        this->lastReward = 0;
        this->lastMapReward = 0;
        this->deathPenalty = 0;
        this->livingReward = 0;
        this->summaryReward = 0;
        this->lastStateNumber = 0;
        this->gameMode = PLAYER;

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

            if(this->availableButtons.size() == 0) {
                //Basic action set
                this->availableButtons.push_back(ATTACK);
                this->availableButtons.push_back(USE);
                this->availableButtons.push_back(JUMP);
                this->availableButtons.push_back(CROUCH);
                this->availableButtons.push_back(SPEED);

                this->availableButtons.push_back(MOVE_RIGHT);
                this->availableButtons.push_back(MOVE_LEFT);
                this->availableButtons.push_back(MOVE_BACKWARD);
                this->availableButtons.push_back(MOVE_FORWARD);
                this->availableButtons.push_back(TURN_RIGHT);
                this->availableButtons.push_back(TURN_LEFT);
            }

            this->lastAction.resize(this->availableButtons.size());

            if(this->gameMode == SPECTATOR){
                this->doomController->setAllowDoomInput(true);
            }

            try {
                this->running = this->doomController->init();

                this->doomController->disableAllButtons();
                for (int i = 0; i < this->availableButtons.size(); ++i) {
                    this->doomController->setButtonAvailable(this->availableButtons[i], true);
                }

                this->state.vars.resize(this->stateAvailableVars.size());

                this->lastReward = 0;
                this->lastMapReward = 0;
                this->lastStateNumber = 0;

                this->updateState();
            }
            catch(const Exception &e){ throw; }

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

        this->state.number = this->doomController->getMapTic();
        this->state.imageBuffer = this->doomController->getScreen();
        
        this->lastReward = 0.0;
        this->lastMapReward = 0.0;
        this->summaryReward = 0.0;
    }

    void DoomGame::setNextAction(std::vector<int> &actions) {

        if (!this->isRunning()) throw DoomIsNotRunningException();

        try {
            for (int i = 0; i < this->availableButtons.size(); ++i) {
                this->lastAction[i] = actions[i];
                this->doomController->setButtonState(this->availableButtons[i], actions[i]);
            }
        }
        catch (...) { throw SharedMemoryException(); }
    }

    void DoomGame::advanceAction() {
        this->advanceAction(true, 1);
    }

    void DoomGame::advanceAction(bool updateState, bool renderOnly) {
        this->advanceAction(updateState, renderOnly, 1);
    }

    void DoomGame::advanceAction(bool updateState, bool renderOnly, unsigned int tics) {

        if (!this->isRunning()) throw DoomIsNotRunningException();

        try {
            if(this->gameMode == PLAYER) this->doomController->tics(tics, updateState || renderOnly);
            else if(this->gameMode == SPECTATOR) this->doomController->realTimeTics(tics, updateState || renderOnly);
        }
        catch(const Exception &e){ throw; }

        if(updateState) this->updateState();
    }

    float DoomGame::makeAction(std::vector<int> &actions){
        this->setNextAction(actions);
        this->advanceAction();
        return this->getLastReward();
    }

    void DoomGame::updateState(){
        try {
            float reward = 0;
            float mapReward = DoomFixedToFloat(this->doomController->getMapReward());
            reward = (mapReward - this->lastMapReward);
            int liveTime = this->doomController->getMapTic() - this->lastStateNumber;
            reward += (liveTime > 0 ? liveTime : 0) * this->livingReward;
            if (this->doomController->isPlayerDead()) reward -= this->deathPenalty;

            this->lastMapReward = mapReward;
            this->summaryReward += reward;
            this->lastReward = reward;

            /* Updates vars */
            for (int i = 0; i < this->stateAvailableVars.size(); ++i) {
                this->state.vars[i] = this->doomController->getGameVar(this->stateAvailableVars[i]);
            }

            /* Update float rgb image */
            this->state.number = this->doomController->getMapTic();
            this->state.imageBuffer = this->doomController->getScreen();

            this->lastStateNumber = this->state.number;

            if (this->gameMode == SPECTATOR) {
                //Update last action
                for (int i = 0; i < this->availableButtons.size(); ++i) {
                    this->lastAction[i] = this->doomController->getButtonState(this->availableButtons[i]);
                }
            }
        }
        catch (...) { throw SharedMemoryException(); }
    }

    DoomGame::State DoomGame::getState() { return this->state; }

    std::vector<int> DoomGame::getLastAction() { return this->lastAction; }

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
        if (!this->running && std::find(this->availableButtons.begin(), this->availableButtons.end(), button) ==
            this->availableButtons.end()) {
            this->availableButtons.push_back(button);
        }
    }

    void DoomGame::addAvailableButton(Button button, int maxValue) {
        if (!this->running && std::find(this->availableButtons.begin(), this->availableButtons.end(), button) ==
                              this->availableButtons.end()) {
            this->availableButtons.push_back(button);
            this->doomController->setButtonMaxValue(button, maxValue);
        }
    }

    void DoomGame::clearAvailableButtons(){
        if(!this->running) this->availableButtons.clear();
    }

    int DoomGame::getAvailableButtonsSize() {
        return this->availableButtons.size();
    }

    void DoomGame::setButtonMaxValue(Button button, int maxValue){
        this->doomController->setButtonMaxValue(button, maxValue);
    }

    void DoomGame::addStateAvailableVar(GameVar var) {
        if (!this->running && std::find(this->stateAvailableVars.begin(), this->stateAvailableVars.end(), var) ==
            this->stateAvailableVars.end()) {
            this->stateAvailableVars.push_back(var);
        }
    }

    void DoomGame::clearStateAvailableVars() {
        if(!this->running) this->stateAvailableVars.clear();
    }

    int DoomGame::getStateAvailableVarsSize() {
        return this->stateAvailableVars.size();
    }

    void DoomGame::addCustomGameArg(std::string arg){
        this->doomController->addCustomArg(arg);
    }

    void DoomGame::clearCustomGameArgs(){
        this->doomController->clearCustomArgs();
    }

    void DoomGame::sendGameCommand(std::string cmd){
        this->doomController->sendCommand(cmd);
    }

    uint8_t * const DoomGame::getGameScreen(){
        this->doomController->getScreen();
    }

    GameMode DoomGame::getGameMode(){ return this->gameMode; };
    void DoomGame::setGameMode(GameMode mode){ if (!this->running) this->gameMode = mode; }

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

    unsigned int DoomGame::getSeed(){ return this->doomController->getSeed(); }
    void DoomGame::setSeed(unsigned int seed){ this->doomController->setSeed(seed); }

    void DoomGame::setAutoNewEpisode(bool set) { this->doomController->setAutoMapRestart(set); }
    void DoomGame::setNewEpisodeOnTimeout(bool set) { this->doomController->setAutoMapRestartOnTimeout(set); }
    void DoomGame::setNewEpisodeOnPlayerDeath(bool set) { this->doomController->setAutoMapRestartOnTimeout(set); }
    void DoomGame::setNewEpisodeOnMapEnd(bool set) { this->doomController->setAutoMapRestartOnMapEnd(set); }

    unsigned int DoomGame::getEpisodeStartTime(){ return this->doomController->getMapStartTime(); }
    void DoomGame::setEpisodeStartTime(unsigned int tics){
        this->doomController->setMapStartTime(tics);
    }

    unsigned int DoomGame::getEpisodeTimeout(){ return this->doomController->getMapTimeout(); }
    void DoomGame::setEpisodeTimeout(unsigned int tics) {
        this->doomController->setMapTimeout(tics);
    }

    float DoomGame::getLivingReward() { return this->livingReward; }
    void DoomGame::setLivingReward(float livingReward) { this->livingReward = livingReward; }

    float DoomGame::getDeathPenalty() { return this->deathPenalty; }
    void DoomGame::setDeathPenalty(float deathPenalty) { this->deathPenalty = deathPenalty; }

    float DoomGame::getLastReward(){ return this->lastReward; }
    float DoomGame::getSummaryReward() { return this->summaryReward; }

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
    void DoomGame::setVisibleWindow(bool visibility) {
        this->doomController->setNoXServer(!visibility);
        this->doomController->setWindowHidden(!visibility);
    }
    void DoomGame::setDisabledConsole(bool noConsole) {
        this->doomController->setNoConsole(noConsole);
    }
    int DoomGame::getScreenWidth() { return this->doomController->getScreenWidth(); }
    int DoomGame::getScreenHeight() { return this->doomController->getScreenHeight(); }
    int DoomGame::getScreenChannels() { return this->doomController->getScreenChannels(); }
    size_t DoomGame::getScreenPitch() { return this->doomController->getScreenPitch(); }
    size_t DoomGame::getScreenSize() { return this->doomController->getScreenSize(); }
    ScreenFormat DoomGame::getScreenFormat() { return this->doomController->getScreenFormat(); }

}
