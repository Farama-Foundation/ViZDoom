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

#include "ViZDoomGame.h"

#include "ViZDoomConfigLoader.h"
#include "ViZDoomController.h"
#include "ViZDoomExceptions.h"
#include "ViZDoomPathHelpers.h"
#include "ViZDoomUtilities.h"

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp> // for reading the shared object/dll path


namespace vizdoom {


    DoomGame::DoomGame() {
        this->running = false;
        this->lastReward = 0;
        this->lastMapReward = 0;
        this->deathPenalty = 0;
        this->livingReward = 0;
        this->summaryReward = 0;
        this->lastMapTic = 0;
        this->nextStateNumber = 1;
        this->mode = PLAYER;

        this->state = nullptr;
        this->doomController = new DoomController();
    }

    DoomGame::~DoomGame() {
        this->close();
        delete this->doomController;
    }

    bool DoomGame::init() {
        if (!this->isRunning()) {

            std::string cfgOverrideFile = "./_vizdoom.cfg";
            if (fileExists(cfgOverrideFile)) loadConfig(cfgOverrideFile);

            this->lastAction.resize(this->availableButtons.size());

            this->doomController->setAllowDoomInput(this->mode == SPECTATOR || this->mode == ASYNC_SPECTATOR);
            this->doomController->setRunDoomAsync(this->mode == ASYNC_PLAYER || this->mode == ASYNC_SPECTATOR);

            try {
                this->running = this->doomController->init();

                this->doomController->disableAllButtons();
                for (unsigned int i = 0; i < this->availableButtons.size(); ++i) {
                    this->doomController->setButtonAvailable(this->availableButtons[i], true);
                }

                this->lastMapTic = 0;
                this->nextStateNumber = 1;

                this->updateState();

                //this->lastMapReward = 0;
                this->lastReward = 0;
                this->summaryReward = 0;

            }
            catch (...) { throw; }

            return running;
        } else return false;
    }

    void DoomGame::close() {
        if (this->isRunning()) {
            this->doomController->close();
            this->lastAction.clear();

            this->state = nullptr;

            this->running = false;
        }
    }

    bool DoomGame::isRunning() {
        return this->running && this->doomController->isDoomRunning();
    }

    void DoomGame::newEpisode(std::string filePath) {

        if (!this->isRunning()) throw ViZDoomIsNotRunningException();

        if (filePath.length()) this->doomController->restartMap(filePath);
        else this->doomController->restartMap();

        this->resetState();
    }

    void DoomGame::replayEpisode(std::string filePath, unsigned int player) {

        if (!this->isRunning()) throw ViZDoomIsNotRunningException();

        this->doomController->playDemo(filePath, player);
        this->resetState();
    }

    void DoomGame::setAction(std::vector<int> const &actions) {

        if (!this->isRunning()) throw ViZDoomIsNotRunningException();

        for (unsigned int i = 0; i < this->availableButtons.size(); ++i) {
            if (i < actions.size()) {
                this->lastAction[i] = actions[i];

            } else {
                this->lastAction[i] = 0;
            }
            this->doomController->setButtonState(this->availableButtons[i], this->lastAction[i]);
        }
    }

    void DoomGame::advanceAction(unsigned int tics, bool updateState) {
        if (!this->isRunning()) throw ViZDoomIsNotRunningException();
        // TODO maybe set lastReward to 0 if finished?
        if (this->doomController->isTicPossible()) {
            try {
                this->doomController->tics(tics, updateState);
            }
            catch (...) { throw; }

            if (updateState) this->updateState();
        }
    }

    double DoomGame::makeAction(std::vector<int> const &actions, unsigned int tics) {
        this->setAction(actions);
        this->advanceAction(tics);
        return this->getLastReward();
    }

    void DoomGame::resetState() {
        this->lastMapTic = 0;
        this->nextStateNumber = 1;

        this->updateState();

        //this->lastMapReward = 0;
        this->lastReward = 0;
        this->summaryReward = 0;
    }

    void DoomGame::updateState() {

        /* Update reward */
        double reward = 0;
        double mapReward = doomFixedToDouble(this->doomController->getMapReward());
        reward = mapReward - this->lastMapReward;
        int liveTime = this->doomController->getMapLastTic() - this->lastMapTic;
        reward += (liveTime > 0 ? liveTime : 0) * this->livingReward;
        if (this->doomController->isPlayerDead()) reward -= this->deathPenalty;

        this->lastMapReward = mapReward;
        this->summaryReward += reward;
        this->lastReward = reward;

        this->lastMapTic = this->doomController->getMapTic();

        /* Update state */
        if (!this->isEpisodeFinished()) {
            this->state = std::make_shared<GameState>();
            this->state->number = this->nextStateNumber++;

            this->state->gameVariables.resize(this->availableGameVariables.size());

            /* Updates vars */
            for (unsigned int i = 0; i < this->availableGameVariables.size(); ++i) {
                this->state->gameVariables[i] =
                        this->doomController->getGameVariable(this->availableGameVariables[i]);
            }

            /* Update buffers */
            int channels = this->getScreenChannels();
            int width = this->getScreenWidth();
            int height = this->getScreenHeight();

            size_t graySize = static_cast<size_t>(width * height);
            size_t colorSize = graySize * channels;

            uint8_t *buf = this->doomController->getScreenBuffer();
            this->state->screenBuffer = std::make_shared<std::vector<uint8_t>>(buf, buf + colorSize);

            if (this->doomController->isDepthBufferEnabled()) {
                buf = this->doomController->getDepthBuffer();
                this->state->depthBuffer = std::make_shared<std::vector<uint8_t>>(buf, buf + graySize);
            } else this->state->depthBuffer = nullptr;

            if (this->doomController->isLabelsEnabled()) {
                buf = this->doomController->getLabelsBuffer();
                this->state->labelsBuffer = std::make_shared<std::vector<uint8_t>>(buf, buf + graySize);
            } else this->state->labelsBuffer = nullptr;

            if (this->doomController->isAutomapEnabled()) {
                buf = this->doomController->getAutomapBuffer();
                this->state->automapBuffer = std::make_shared<std::vector<uint8_t>>(buf, buf + colorSize);
            } else this->state->automapBuffer = nullptr;

            /* Update labels */
            this->state->labels.clear();
            for (unsigned int i = 0; i < this->doomController->getGameState()->LABEL_COUNT; ++i) {
                Label label;
                label.objectId = this->doomController->getGameState()->LABEL[i].objectId;
                label.objectName = std::string(this->doomController->getGameState()->LABEL[i].objectName);
                label.value = this->doomController->getGameState()->LABEL[i].value;
                label.objectPositionX = this->doomController->getGameState()->LABEL[i].objectPosition[0];
                label.objectPositionY = this->doomController->getGameState()->LABEL[i].objectPosition[1];
                label.objectPositionZ = this->doomController->getGameState()->LABEL[i].objectPosition[2];
                this->state->labels.push_back(label);
            }
        } else this->state = nullptr;

        /* Update last action */
        for (unsigned int i = 0; i < this->availableButtons.size(); ++i) {
            this->lastAction[i] = this->doomController->getButtonState(this->availableButtons[i]);
        }
    }

    std::shared_ptr<GameState> DoomGame::getState() { return this->state; }

    std::vector<int> DoomGame::getLastAction() { return this->lastAction; }

    bool DoomGame::isNewEpisode() {
        if (!this->isRunning()) throw ViZDoomIsNotRunningException();
        return this->doomController->isMapFirstTic();
    }

    bool DoomGame::isEpisodeFinished() {
        if (!this->isRunning()) throw ViZDoomIsNotRunningException();
        return !this->doomController->isTicPossible();
    }

    bool DoomGame::isPlayerDead() {
        if (!this->isRunning()) throw ViZDoomIsNotRunningException();
        return this->doomController->isPlayerDead();
    }

    void DoomGame::respawnPlayer() {
        if (!this->isRunning()) throw ViZDoomIsNotRunningException();

        this->doomController->respawnPlayer();
        this->updateState();
        this->lastReward = 0;
    }

    std::vector<Button> DoomGame::getAvailableButtons() {
        return this->availableButtons;
    }

    void DoomGame::setAvailableButtons(std::vector<Button> buttons) {
        this->clearAvailableButtons();
        for(auto i : buttons) this->addAvailableButton(i);
    }

    void DoomGame::addAvailableButton(Button button, unsigned int maxValue) {
        if (!this->isRunning() && std::find(this->availableButtons.begin(),
                                            this->availableButtons.end(), button) == this->availableButtons.end()) {
            this->availableButtons.push_back(button);
        }
        if(maxValue != -1) this->doomController->setButtonMaxValue(button, maxValue);
    }

    void DoomGame::clearAvailableButtons() {
        if (!this->isRunning()) this->availableButtons.clear();
    }

    size_t DoomGame::getAvailableButtonsSize() {
        return this->availableButtons.size();
    }

    void DoomGame::setButtonMaxValue(Button button, unsigned int maxValue) {
        this->doomController->setButtonMaxValue(button, maxValue);
    }

    int DoomGame::getButtonMaxValue(Button button) {
        return this->doomController->getButtonMaxValue(button);
    }

    std::vector<GameVariable> DoomGame::getAvailableGameVariables(){
        return this->availableGameVariables;
    }

    void DoomGame::setAvailableGameVariables(std::vector<GameVariable> gameVariables){
        this->clearAvailableGameVariables();
        for(auto i : gameVariables) this->addAvailableGameVariable(i);
    }

    void DoomGame::addAvailableGameVariable(GameVariable var) {
        if (!this->isRunning() &&
            std::find(this->availableGameVariables.begin(), this->availableGameVariables.end(), var)
            == this->availableGameVariables.end()) {
            this->availableGameVariables.push_back(var);
        }
    }

    void DoomGame::clearAvailableGameVariables() {
        if (!this->isRunning()) this->availableGameVariables.clear();
    }

    size_t DoomGame::getAvailableGameVariablesSize() {
        return this->availableGameVariables.size();
    }

    void DoomGame::addGameArgs(std::string args) {
        if (args.length() != 0) {
            std::vector<std::string> _args;
            b::split(_args, args, b::is_any_of("\t\n "));
            for (unsigned int i = 0; i < _args.size(); ++i) {
                if (_args[i].length() > 0) this->doomController->addCustomArg(_args[i]);
            }
        }
    }

    void DoomGame::clearGameArgs() {
        this->doomController->clearCustomArgs();
    }

    void DoomGame::sendGameCommand(std::string cmd) {
        this->doomController->sendCommand(cmd);
    }

    Mode DoomGame::getMode() { return this->mode; };

    void DoomGame::setMode(Mode mode) { if (!this->isRunning()) this->mode = mode; }

    unsigned int DoomGame::getTicrate() { return this->doomController->getTicrate(); }

    void DoomGame::setTicrate(unsigned int ticrate) { this->doomController->setTicrate(ticrate); }

    double DoomGame::getGameVariable(GameVariable variable){
        if(!this->isRunning()) throw ViZDoomIsNotRunningException();
        return this->doomController->getGameVariable(variable);
    }

    void DoomGame::setViZDoomPath(std::string filePath) { this->doomController->setExePath(filePath); }

    void DoomGame::setDoomGamePath(std::string filePath) { this->doomController->setIwadPath(filePath); }

    void DoomGame::setDoomScenarioPath(std::string filePath) { this->doomController->setFilePath(filePath); }

    void DoomGame::setDoomMap(std::string map) {
        this->doomController->setMap(map);
        if (this->isRunning()) this->resetState();
    }

    void DoomGame::setDoomSkill(int skill) { this->doomController->setSkill(skill); }

    void DoomGame::setDoomConfigPath(std::string filePath) { this->doomController->setConfigPath(filePath); }

    unsigned int DoomGame::getSeed() { return this->doomController->getInstanceSeed(); }

    void DoomGame::setSeed(unsigned int seed) { this->doomController->setInstanceSeed(seed); }

    unsigned int DoomGame::getEpisodeStartTime() { return this->doomController->getMapStartTime(); }

    void DoomGame::setEpisodeStartTime(unsigned int tics) { this->doomController->setMapStartTime(tics); }

    unsigned int DoomGame::getEpisodeTimeout() { return this->doomController->getMapTimeout(); }

    void DoomGame::setEpisodeTimeout(unsigned int tics) { this->doomController->setMapTimeout(tics); }

    unsigned int DoomGame::getEpisodeTime() { return this->doomController->getMapTic(); }

    double DoomGame::getLivingReward() { return this->livingReward; }

    void DoomGame::setLivingReward(double livingReward) { this->livingReward = livingReward; }

    double DoomGame::getDeathPenalty() { return this->deathPenalty; }

    void DoomGame::setDeathPenalty(double deathPenalty) { this->deathPenalty = deathPenalty; }

    double DoomGame::getLastReward() { return this->lastReward; }

    double DoomGame::getTotalReward() { return this->summaryReward; }

    void DoomGame::setScreenResolution(ScreenResolution resolution) {
        unsigned int width = 0, height = 0;

#define CASE_RES(w, h) case RES_##w##X##h : width = w; height = h; break;

        switch (resolution) {
            CASE_RES(160, 120)

            CASE_RES(200, 125)
            CASE_RES(200, 150)

            CASE_RES(256, 144)
            CASE_RES(256, 160)
            CASE_RES(256, 192)

            CASE_RES(320, 180)
            CASE_RES(320, 200)
            CASE_RES(320, 240)
            CASE_RES(320, 256)

            CASE_RES(400, 225)
            CASE_RES(400, 250)
            CASE_RES(400, 300)

            CASE_RES(512, 288)
            CASE_RES(512, 320)
            CASE_RES(512, 384)

            CASE_RES(640, 360)
            CASE_RES(640, 400)
            CASE_RES(640, 480)

            CASE_RES(800, 450)
            CASE_RES(800, 500)
            CASE_RES(800, 600)

            CASE_RES(1024, 576)
            CASE_RES(1024, 640)
            CASE_RES(1024, 768)

            CASE_RES(1280, 720)
            CASE_RES(1280, 800)
            CASE_RES(1280, 960)
            CASE_RES(1280, 1024)

            CASE_RES(1400, 787)
            CASE_RES(1400, 875)
            CASE_RES(1400, 1050)

            CASE_RES(1600, 900)
            CASE_RES(1600, 1000)
            CASE_RES(1600, 1200)

            CASE_RES(1920, 1080)
        }
        this->doomController->setScreenResolution(width, height);
    }

    void DoomGame::setScreenFormat(ScreenFormat format) { this->doomController->setScreenFormat(format); }

    bool DoomGame::isDepthBufferEnabled() { return this->doomController->isDepthBufferEnabled(); }

    void DoomGame::setDepthBufferEnabled(bool depthBuffer) { this->doomController->setDepthBufferEnabled(depthBuffer); }

    bool DoomGame::isLabelsBufferEnabled() { return this->doomController->isLabelsEnabled(); }

    void DoomGame::setLabelsBufferEnabled(bool lebelsBuffer) { this->doomController->setLabelsEnabled(lebelsBuffer); }

    bool DoomGame::isAutomapBufferEnabled() { return this->doomController->isAutomapEnabled(); }

    void DoomGame::setAutomapBufferEnabled(bool automapBuffer) {
        this->doomController->setAutomapEnabled(automapBuffer);
    }

    void DoomGame::setAutomapMode(AutomapMode mode) { this->doomController->setAutomapMode(mode); }

    void DoomGame::setAutomapRotate(bool rotate) { this->doomController->setAutomapRotate(rotate); }

    void DoomGame::setAutomapRenderTextures(bool textures) { this->doomController->setAutomapRenderTextures(textures); }

    void DoomGame::setRenderHud(bool hud) { this->doomController->setRenderHud(hud); }

    void DoomGame::setRenderMinimalHud(bool minimalHud) { this->doomController->setRenderMinimalHud(minimalHud); }

    void DoomGame::setRenderWeapon(bool weapon) { this->doomController->setRenderWeapon(weapon); }

    void DoomGame::setRenderCrosshair(bool crosshair) { this->doomController->setRenderCrosshair(crosshair); }

    void DoomGame::setRenderDecals(bool decals) { this->doomController->setRenderDecals(decals); }

    void DoomGame::setRenderParticles(bool particles) { this->doomController->setRenderParticles(particles); }

    void DoomGame::setRenderEffectsSprites(bool sprites) { this->doomController->setRenderEffectsSprites(sprites); }

    void DoomGame::setRenderMessages(bool messages) { this->doomController->setRenderMessages(messages); }
    void DoomGame::setRenderCorpses(bool corpses) { this->doomController->setRenderCorpses(corpses); }

    void DoomGame::setWindowVisible(bool visibility) {
        this->doomController->setNoXServer(!visibility);
        this->doomController->setWindowHidden(!visibility);
    }

    void DoomGame::setConsoleEnabled(bool console) { this->doomController->setNoConsole(!console); }

    void DoomGame::setSoundEnabled(bool sound) { this->doomController->setNoSound(!sound); }

    int DoomGame::getScreenWidth() { return this->doomController->getScreenWidth(); }

    int DoomGame::getScreenHeight() { return this->doomController->getScreenHeight(); }

    int DoomGame::getScreenChannels() { return this->doomController->getScreenChannels(); }

    size_t DoomGame::getScreenPitch() { return this->doomController->getScreenPitch(); }

    size_t DoomGame::getScreenSize() { return this->doomController->getScreenSize(); }

    ScreenFormat DoomGame::getScreenFormat() { return this->doomController->getScreenFormat(); }

    bool DoomGame::loadConfig(std::string filePath) {
        ConfigLoader configLoader(this);
        return configLoader.load(filePath);
    }

}




