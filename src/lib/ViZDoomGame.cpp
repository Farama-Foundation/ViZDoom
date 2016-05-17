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
#include "ViZDoomController.h"
#include "ViZDoomExceptions.h"
#include "ViZDoomUtilities.h"

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/trim_all.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>
#include <climits>
#include <cstdlib>
#include <ctime>
#include <fstream>

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
        this->seed = (unsigned int)(time(NULL) % UINT_MAX) ;
        this->mode = PLAYER;


        this->doomController = new DoomController();
    }

    DoomGame::~DoomGame() {
        this->close();
        delete this->doomController;
    }

    bool DoomGame::init() {
        if (!this->running) {

            this->lastAction.resize(this->availableButtons.size());

            if(this->mode == SPECTATOR || this->mode == ASYNC_SPECTATOR){
                this->doomController->setAllowDoomInput(true);
            } else {
                this->doomController->setAllowDoomInput(false);
            }

            if(this->mode == ASYNC_PLAYER || this->mode == ASYNC_SPECTATOR){
                this->doomController->setRunDoomAsync(true);
            } else {
                this->doomController->setRunDoomAsync(false);
            }

            try {
                this->running = this->doomController->init();

                this->doomController->disableAllButtons();
                for (unsigned int i = 0; i < this->availableButtons.size(); ++i) {
                    this->doomController->setButtonAvailable(this->availableButtons[i], true);
                }

                this->state.gameVariables.resize(this->availableGameVariables.size());

                this->lastMapTic = 0;
                this->nextStateNumber = 1;

                this->updateState();

                //this->lastMapReward = 0;
                this->lastReward = 0;
                this->summaryReward = 0;

            }
            catch(...){ throw; }

            return running;
        }
        else return false;
    }

    void DoomGame::close() {
        if (this->running) {
            this->doomController->close();
            this->state.gameVariables.clear();
            this->lastAction.clear();

            this->running = false;
        }
    }

    bool DoomGame::isRunning(){
        return this->running && this->doomController->isDoomRunning();
    }

    void DoomGame::newEpisode() {

        if(!this->isRunning()) throw ViZDoomIsNotRunningException();

        this->doomController->setRngSeed((unsigned int)(rand() % UINT_MAX));
        this->doomController->restartMap();
        this->doomController->clearRngSeed();

        this->lastMapTic = 0;
        this->nextStateNumber = 1;

        this->updateState();

        //this->lastMapReward = 0;
        this->lastReward = 0;
        this->summaryReward = 0;
    }

    void DoomGame::setAction(std::vector<int> const &actions) {

        if (!this->isRunning()) throw ViZDoomIsNotRunningException();
        try {
            for (unsigned int i = 0; i < this->availableButtons.size(); ++i) {
                if(i < actions.size()){
                    this->lastAction[i] = actions[i];

                }
                else{
                    this->lastAction[i] = 0;
                }
                this->doomController->setButtonState(this->availableButtons[i], this->lastAction[i]);
            }
        }
        catch (...) { throw SharedMemoryException(); }
    }

    void DoomGame::advanceAction() {
        this->advanceAction(1, true, true);
    }

    void DoomGame::advanceAction(unsigned int tics) {
        this->advanceAction(tics, true, true);
    }

    void DoomGame::advanceAction(unsigned int tics, bool updateState, bool renderOnly) {

        if (!this->isRunning()) throw ViZDoomIsNotRunningException();

        try {
            this->doomController->tics(tics, updateState || renderOnly);
        }
        catch(...){ throw; }

        if(updateState) this->updateState();
    }

    double DoomGame::makeAction(std::vector<int> const &actions){
        this->setAction(actions);
        this->advanceAction();
        return this->getLastReward();
    }

    double DoomGame::makeAction(std::vector<int> const &actions, unsigned int tics){
        this->setAction(actions);
        this->advanceAction(tics);
        return this->getLastReward();
    }

    void DoomGame::updateState(){
        try {

            this->state.number = this->nextStateNumber++;

            double reward = 0;
            double mapReward = DoomFixedToDouble(this->doomController->getMapReward());
            reward = mapReward - this->lastMapReward;
            int liveTime = this->doomController->getMapLastTic() - this->lastMapTic;
            reward += (liveTime > 0 ? liveTime : 0) * this->livingReward;
            if (this->doomController->isPlayerDead()) reward -= this->deathPenalty;

            this->lastMapReward = mapReward;
            this->summaryReward += reward;
            this->lastReward = reward;

            this->lastMapTic = this->doomController->getMapTic();

            /* Updates vars */
            for (unsigned int i = 0; i < this->availableGameVariables.size(); ++i) {
                this->state.gameVariables[i] = this->doomController->getGameVariable(this->availableGameVariables[i]);
            }

            /* Update float rgb image */
            this->state.imageBuffer = this->doomController->getScreen();

            //Update last action
            for (unsigned int i = 0; i < this->availableButtons.size(); ++i) {
                this->lastAction[i] = this->doomController->getButtonState(this->availableButtons[i]);
            }
        }
        catch (...) { throw SharedMemoryException(); }
    }

    GameState DoomGame::getState() { return this->state; }

    std::vector<int> DoomGame::getLastAction() { return this->lastAction; }

    bool DoomGame::isNewEpisode() {
        if(!this->isRunning()) throw ViZDoomIsNotRunningException();

        return this->doomController->isMapFirstTic();
    }

    bool DoomGame::isEpisodeFinished() {
        if(!this->isRunning()) throw ViZDoomIsNotRunningException();

        return !this->doomController->isTicPossible();
    }

    bool DoomGame::isPlayerDead() {
        if(!this->isRunning()) throw ViZDoomIsNotRunningException();
        return this->doomController->isPlayerDead();
    }

    void DoomGame::respawnPlayer(){
        if(!this->isRunning()) throw ViZDoomIsNotRunningException();

        this->doomController->respawnPlayer();
        this->updateState();
        this->lastReward = 0;
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

    int DoomGame::getButtonMaxValue(Button button){
        return this->doomController->getButtonMaxValue(button);
    }

    void DoomGame::addAvailableGameVariable(GameVariable var) {
        if (!this->running && std::find(this->availableGameVariables.begin(), this->availableGameVariables.end(), var) ==
            this->availableGameVariables.end()) {
            this->availableGameVariables.push_back(var);
        }
    }

    void DoomGame::clearAvailableGameVariables() {
        if(!this->running) this->availableGameVariables.clear();
    }

    int DoomGame::getAvailableGameVariablesSize() {
        return this->availableGameVariables.size();
    }

    void DoomGame::addGameArgs(std::string args){
        if (args.length() != 0) {
            std::vector<std::string> _args;
            b::split(_args, args, b::is_any_of("\t\n "));
            for (unsigned int i = 0; i < _args.size(); ++i) {
                if(_args[i].length() > 0) this->doomController->addCustomArg(_args[i]);
            }
        }
    }

    void DoomGame::clearGameArgs(){
        this->doomController->clearCustomArgs();
    }

    void DoomGame::sendGameCommand(std::string cmd){
        this->doomController->sendCommand(cmd);
    }

    uint8_t * const DoomGame::getGameScreen(){
        return this->doomController->getScreen();
    }

    Mode DoomGame::getMode(){ return this->mode; };
    void DoomGame::setMode(Mode mode){ if (!this->running) this->mode = mode; }

    int DoomGame::getGameVariable(GameVariable var){
        if(!this->isRunning()) throw ViZDoomIsNotRunningException();

        return this->doomController->getGameVariable(var);
    }

    void DoomGame::setViZDoomPath(std::string path) { this->doomController->setExePath(path); }
    void DoomGame::setDoomGamePath(std::string path) { this->doomController->setIwadPath(path); }
    void DoomGame::setDoomScenarioPath(std::string path) { this->doomController->setFilePath(path); }
    void DoomGame::setDoomMap(std::string map) { this->doomController->setMap(map); }
    void DoomGame::setDoomSkill(int skill) {
        this->doomController->setSkill(skill); 
    }
    void DoomGame::setDoomConfigPath(std::string path) { this->doomController->setConfigPath(path); }

    unsigned int DoomGame::getSeed(){ return this->seed; }

    void DoomGame::setSeed(unsigned int seed){
        this->seed = seed;
        srand(this->seed);
    }

    unsigned int DoomGame::getEpisodeStartTime(){ return this->doomController->getMapStartTime(); }
    void DoomGame::setEpisodeStartTime(unsigned int tics){
        this->doomController->setMapStartTime(tics);
    }

    unsigned int DoomGame::getEpisodeTimeout(){ return this->doomController->getMapTimeout(); }
    void DoomGame::setEpisodeTimeout(unsigned int tics) {
        this->doomController->setMapTimeout(tics);
    }

    unsigned int DoomGame::getEpisodeTime(){ return this->doomController->getMapTic(); }

    double DoomGame::getLivingReward() { return this->livingReward; }
    void DoomGame::setLivingReward(double livingReward) { this->livingReward = livingReward; }

    double DoomGame::getDeathPenalty() { return this->deathPenalty; }
    void DoomGame::setDeathPenalty(double deathPenalty) { this->deathPenalty = deathPenalty; }

    double DoomGame::getLastReward(){ return this->lastReward; }
    double DoomGame::getTotalReward() { return this->summaryReward; }

    void DoomGame::setScreenResolution(ScreenResolution resolution) {
        unsigned int width = 0, height = 0;

        #define CASE_RES(w, h) case RES_##w##X##h : width = w; height = h; break;
        switch(resolution){
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
    void DoomGame::setRenderHud(bool hud) { this->doomController->setRenderHud(hud); }
    void DoomGame::setRenderWeapon(bool weapon) { this->doomController->setRenderWeapon(weapon); }
    void DoomGame::setRenderCrosshair(bool crosshair) { this->doomController->setRenderCrosshair(crosshair); }
    void DoomGame::setRenderDecals(bool decals) { this->doomController->setRenderDecals(decals); }
    void DoomGame::setRenderParticles(bool particles) { this->doomController->setRenderParticles(particles); }
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


    /* Code used for parsing the config file. */
    //TODO warnings, refactoring, comments
    bool DoomGame::StringToBool(std::string boolString){
        if(boolString == "true" || boolString == "1")   return true;
        if(boolString == "false" || boolString == "0")  return false;

        throw std::exception();
    }

    unsigned int DoomGame::StringToUint(std::string str)
    {
        unsigned int value = boost::lexical_cast<unsigned int>(str);
        if(str[0] == '-') throw boost::bad_lexical_cast();
        return value;
    }

    ScreenResolution DoomGame::StringToResolution(std::string str){
        if(str == "res_160x120")    return RES_160X120;

        if(str == "res_200x125")    return RES_200X125;
        if(str == "res_200x150")    return RES_200X150;

        if(str == "res_256x144")    return RES_256X144;
        if(str == "res_256x160")    return RES_256X160;
        if(str == "res_256x192")    return RES_256X192;

        if(str == "res_320x180")    return RES_320X180;
        if(str == "res_320x200")    return RES_320X200;
        if(str == "res_320x240")    return RES_320X240;
        if(str == "res_320x256")    return RES_320X256;

        if(str == "res_400x225")    return RES_400X225;
        if(str == "res_400x250")    return RES_400X250;
        if(str == "res_400x300")    return RES_400X300;

        if(str == "res_512x288")    return RES_512X288;
        if(str == "res_512x320")    return RES_512X320;
        if(str == "res_512x384")    return RES_512X384;

        if(str == "res_640x360")    return RES_640X400;
        if(str == "res_640x400")    return RES_640X400;
        if(str == "res_640x480")    return RES_640X480;

        if(str == "res_800x450")    return RES_800X450;
        if(str == "res_800x500")    return RES_800X500;
        if(str == "res_800x600")    return RES_800X600;

        if(str == "res_1024x576")   return RES_1024X576;
        if(str == "res_1024x640")   return RES_1024X640;
        if(str == "res_1024x768")   return RES_1024X768;

        if(str == "res_1280x720")   return RES_1280X720;
        if(str == "res_1280x800")   return RES_1280X800;
        if(str == "res_1280x960")   return RES_1280X960;
        if(str == "res_1280x1024")  return RES_1280X1024;

        if(str == "res_1400x787")   return RES_1400X787;
        if(str == "res_1400x875")   return RES_1400X875;
        if(str == "res_1400x1050")  return RES_1400X1050;

        if(str == "res_1600x900")   return RES_1600X900;
        if(str == "res_1600x1000")  return RES_1600X1000;
        if(str == "res_1600x1200")  return RES_1600X1200;

        if(str == "res_1920x1080")  return RES_1920X1080;

        throw std::exception();
    }
        
    ScreenFormat DoomGame::StringToFormat(std::string str){
        if(str == "crcgcb")             return CRCGCB;
        if(str == "crcgcbzb")           return CRCGCBDB;
        if(str == "rgb24")              return RGB24;
        if(str == "rgba32")             return RGBA32;
        if(str == "argb32")             return ARGB32;
        if(str == "cbcgcr")             return CBCGCR;
        if(str == "cbcgcrzb")           return CBCGCRDB;
        if(str == "bgr24")              return BGR24;
        if(str == "bgra32")             return BGRA32;
        if(str == "abgr32")             return ABGR32;
        if(str == "gray8")              return GRAY8;
        if(str == "zbuffer8")           return DEPTH_BUFFER8;
        if(str == "doom_256_colors8")   return DOOM_256_COLORS8;

        throw std::exception();
    }
     
    Button DoomGame::StringToButton(std::string str){
        if(str == "attack")         return ATTACK;
        if(str == "use")            return USE;
        if(str == "jump")           return JUMP;
        if(str == "crouch")         return CROUCH;
        if(str == "turn180")        return TURN180;
        if(str == "alattack")       return ALTATTACK;
        if(str == "reload")         return RELOAD;
        if(str == "zoom")           return ZOOM;
        if(str == "speed")          return SPEED;
        if(str == "strafe")         return STRAFE;
        if(str == "move_right")     return MOVE_RIGHT;
        if(str == "move_left")      return MOVE_LEFT;
        if(str == "move_backward")  return MOVE_BACKWARD;
        if(str == "move_forward")   return MOVE_FORWARD;
        if(str == "turn_right")     return TURN_RIGHT;
        if(str == "turn_left")      return TURN_LEFT;
        if(str == "look_up")        return LOOK_UP;
        if(str == "look_down")      return LOOK_DOWN;
        if(str == "move_up")        return MOVE_UP;
        if(str == "move_down")      return MOVE_DOWN;
        if(str == "land")           return LAND;

        if(str == "select_weapon1") return SELECT_WEAPON1;
        if(str == "select_weapon2") return SELECT_WEAPON2;
        if(str == "select_weapon3") return SELECT_WEAPON3;
        if(str == "select_weapon4") return SELECT_WEAPON4;
        if(str == "select_weapon5") return SELECT_WEAPON5;
        if(str == "select_weapon6") return SELECT_WEAPON6;
        if(str == "select_weapon7") return SELECT_WEAPON7;
        if(str == "select_weapon8") return SELECT_WEAPON8;
        if(str == "select_weapon9") return SELECT_WEAPON9;
        if(str == "select_weapon0") return SELECT_WEAPON0;

        if(str == "select_next_weapon")         return SELECT_NEXT_WEAPON;
        if(str == "select_prev_weapon")         return SELECT_PREV_WEAPON;
        if(str == "drop_selected_weapon")       return DROP_SELECTED_WEAPON;
        if(str == "activate_selected_weapon")   return ACTIVATE_SELECTED_ITEM;
        if(str == "select_next_item")           return SELECT_NEXT_ITEM;
        if(str == "select_prev_item")           return SELECT_PREV_ITEM;
        if(str == "drop_selected_item")         return DROP_SELECTED_ITEM;

        if(str == "look_up_down_delta")         return LOOK_UP_DOWN_DELTA;
        if(str == "turn_left_right_delta")      return TURN_LEFT_RIGHT_DELTA;
        if(str == "move_forward_backward_delta")return MOVE_FORWARD_BACKWARD_DELTA;
        if(str == "move_left_right_delta")      return MOVE_LEFT_RIGHT_DELTA;
        if(str == "move_up_down_delta")         return MOVE_UP_DOWN_DELTA;
       
        throw std::exception();
    }

    GameVariable DoomGame::StringToGameVariable(std::string str){
        if(str == "killcount")              return KILLCOUNT;
        if(str == "itemcount")              return ITEMCOUNT;
        if(str == "secretcount")            return SECRETCOUNT;
        if(str == "fragcount")              return FRAGCOUNT;
        if(str == "health")                 return HEALTH;
        if(str == "armor")                  return ARMOR;
        if(str == "dead")                   return DEAD;
        if(str == "on_ground")              return ON_GROUND;
        if(str == "attack_ready")           return ATTACK_READY;
        if(str == "altattack_ready")        return ALTATTACK_READY;
        if(str == "selected_weapon")        return SELECTED_WEAPON;
        if(str == "selected_weapon_ammo")   return SELECTED_WEAPON_AMMO;
         
        if(str == "ammo1")      return AMMO1;
        if(str == "ammo2")      return AMMO2;
        if(str == "ammo3")      return AMMO3;
        if(str == "ammo4")      return AMMO4;
        if(str == "ammo5")      return AMMO5;
        if(str == "ammo6")      return AMMO6;
        if(str == "ammo7")      return AMMO7;
        if(str == "ammo8")      return AMMO8;
        if(str == "ammo9")      return AMMO9;
        if(str == "ammo0")      return AMMO0;

        if(str == "weapon1")    return WEAPON1;
        if(str == "weapon2")    return WEAPON2;
        if(str == "weapon3")    return WEAPON3;
        if(str == "weapon4")    return WEAPON4;
        if(str == "weapon5")    return WEAPON5;
        if(str == "weapon6")    return WEAPON6;
        if(str == "weapon7")    return WEAPON7;
        if(str == "weapon8")    return WEAPON8;
        if(str == "weapon9")    return WEAPON9;
        if(str == "weapon0")    return WEAPON0;

        if(str == "user1")      return USER1;
        if(str == "user2")      return USER2;
        if(str == "user3")      return USER3;
        if(str == "user4")      return USER4;
        if(str == "user5")      return USER5;
        if(str == "user6")      return USER6;
        if(str == "user7")      return USER7;
        if(str == "user8")      return USER8;
        if(str == "user9")      return USER9;
        if(str == "user10")     return USER10;
        if(str == "user11")     return USER11;
        if(str == "user12")     return USER12;
        if(str == "user13")     return USER13;
        if(str == "user14")     return USER14;
        if(str == "user15")     return USER15;
        if(str == "user16")     return USER16;
        if(str == "user17")     return USER17;
        if(str == "user18")     return USER18;
        if(str == "user19")     return USER19;
        if(str == "user20")     return USER20;
        if(str == "user21")     return USER21;
        if(str == "user22")     return USER22;
        if(str == "user23")     return USER23;
        if(str == "user24")     return USER24;
        if(str == "user25")     return USER25;
        if(str == "user26")     return USER26;
        if(str == "user27")     return USER27;
        if(str == "user28")     return USER28;
        if(str == "user29")     return USER29;
        if(str == "user30")     return USER30;

        throw std::exception();
    }

    typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
    bool DoomGame::ParseListProperty(int& line_number, std::string& value, std::ifstream& input, std::vector<std::string>& output){
        using namespace boost::algorithm;
        int start_line = line_number;
    /* Find '{' */
        while(value.empty()){
            if(!input.eof()){
                ++line_number;
                std::getline(input, value);
                trim_all(value);
                if(!value.empty() && value[0]=='#')
                    value="";
            }
            else
                break;
        }
        if(value.empty() || value[0] != '{') return false;
        
        value = value.substr(1);

        /* Find '}' */
        while((value.empty() || value[value.size()-1] != '}') && !input.eof()){
            ++line_number;
            std::string newline;
            std::getline(input, newline);
            trim_all(newline);
            if(!newline.empty() && newline[0]!='#')
                value += std::string(" ") + newline;
        }
        if(value.empty() || value[value.size()-1] != '}') return false;
        
    /* Fill the vector */
        value[value.size() -1] = ' ';
        trim_all(value);
        to_lower(value);
        
        boost::char_separator<char> separator(" ");
        tokenizer tok(value, separator);
        for(tokenizer::iterator it = tok.begin(); it != tok.end(); ++it){
            output.push_back(*it);
        }
        return true;
    }

    bool DoomGame::loadConfig(std::string filename) {
        bool success = true;
        std::ifstream file(filename.c_str());
        
        if(!file.good() )
        {
            throw FileDoesNotExistException(filename);
            //std::cerr<<"WARNING! Loading config from: \""<<filename<<"\" failed. Something's wrong with the file. Check your spelling and permissions.\n";
            return false;
        }
        std::string line;
        int line_number = 0;

    /* Process every line. */
        while(!file.eof())
        {
            ++line_number;
            using namespace boost::algorithm;

            std::getline(file, line);

        /* Ignore empty and comment lines */
            trim_all(line);

            if(line.empty() || line[0] == '#'){
                continue;
            }


        bool append = false; //it looks for +=

        /* Check if '=' is there */
            size_t equals_sign_pos = line.find_first_of('=');
            size_t append_sign_pos = line.find("+=");

            std::string key;
            std::string val;
            std::string raw_val;
            if( append_sign_pos != std::string::npos){
                key = line.substr(0, append_sign_pos);
                val = line.substr(append_sign_pos + 2);
                append = true;
            }
            else if( equals_sign_pos != std::string::npos ){
                key = line.substr(0, equals_sign_pos);
                val = line.substr(equals_sign_pos + 1);
            }
            else
            {
                std::cerr<<"WARNING! Loading config from: \""<<filename<<"\". Syntax erorr in line #"<<line_number<<". Line ignored.\n";
                success = false;
                continue;
            }

            
            raw_val = val;
            trim_all(key);
            trim_all(val);
            std::string original_val = val;
            to_lower(val);
            to_lower(key);
            if(key.empty())
            {
                std::cerr<<"WARNING! Loading config from: \""<<filename<<"\". Empty key in line #"<<line_number<<". Line ignored.\n";
                success = false;
                continue;
            }


        /* Parse enum list properties */

            if(key == "available_buttons" || key == "availablebuttons"){
                std::vector<std::string> str_buttons;
                int start_line = line_number;
                bool parse_success = DoomGame::ParseListProperty(line_number, val, file, str_buttons );
                if(parse_success){
                    unsigned int i = 0;
                    try{
                        std::vector<Button> buttons;
                        for( i = 0; i < str_buttons.size(); ++i ){
                            buttons.push_back(DoomGame::StringToButton(str_buttons[i]));

                        }
                        if (!append)
                            this->clearAvailableButtons();
                        for( i = 0; i < buttons.size(); ++i ){
                            this->addAvailableButton(buttons[i]);
                        }
                    }
                    catch(std::exception){
                        std::cerr<<"WARNING! Loading config from: \""<<filename<<"\". Unsupported value in lines "<<start_line<<"-"<<line_number<<": "<<str_buttons[i]<<". Lines ignored.\n";
                        success = false;
                    }
                }
                else{
                    std::cerr<<"WARNING! Loading config from: \""<<filename<<"\". Syntax error in lines "<<start_line<<"-"<<line_number<<". Lines ignored.\n";
                    success = false;
                }
  
                continue;
            }

            if(key == "available_game_variables" || key == "availablegamevariables"){
                std::vector<std::string> str_variables;
                int start_line = line_number;
                bool parse_success = DoomGame::ParseListProperty(line_number, val, file, str_variables );
                if(parse_success){
                    unsigned int i = 0;
                    try{
                        std::vector<GameVariable> variables;
                        for( i = 0; i < str_variables.size(); ++i ){
                            variables.push_back(DoomGame::StringToGameVariable(str_variables[i]));

                        }
                        if(!append)
                            this->clearAvailableGameVariables();
                        for( i = 0; i < variables.size(); ++i ){
                            this->addAvailableGameVariable(variables[i]);
                        }
                    }
                    catch(std::exception){
                        std::cerr<<"WARNING! Loading config from: \""<<filename<<"\". Unsupported value in lines "<<start_line<<"-"<<line_number<<": "<<str_variables[i]<<". Lines ignored.\n";
                        success = false;
                    }
                }
                else{
                    std::cerr<<"WARNING! Loading config from: \""<<filename<<"\". Syntax error in lines "<<start_line<<"-"<<line_number<<". Lines ignored.\n";
                    success = false;
                }

                continue;
            }           

        /* Parse game args which ae string but enables "+=" */
            if(key == "game_args" || key == "game_args"){
            	if(!append){
            		this->clearGameArgs();
            	}
                this->addGameArgs(original_val);
                continue;
            }
        /* Check if "+=" was not used for non-list property */
            if(append){
                std::cerr<<"WARNING! Loading config from: \""<<filename<<"\". \"+=\" is not supported for non-list properties. Line #"<<line_number<<" ignored.\n";
                success = false;
                continue;
            }

           	
        /* Check if value is not empty */
            if(val.empty())
            {
                std::cerr<<"WARNING! Loading config from: \""<<filename<<"\". Empty value in line #"<<line_number<<". Line ignored.\n";
                success = false;
                continue;
            }
        
        /* Parse int properties */
            try{
                if (key =="seed" || key == "seed"){
                    this->setSeed(StringToUint(val));
                    continue;
                }
                if (key == "episode_timeout" || key == "episodetimeout"){
                    this->setEpisodeTimeout(StringToUint(val));
                    continue;
                }
                if (key == "episode_start_time" || key == "episodestarttime"){
                    this->setEpisodeStartTime(StringToUint(val));
                    continue;
                }
                if (key == "doom_skill" || key == "doomskill"){
                    this->setDoomSkill(StringToUint(val));
                    continue;
                }
            }
            catch(boost::bad_lexical_cast &){
                std::cerr<<"WARNING! Loading config from: \""<<filename<<"\". Unsigned int value expected insted of: "<<raw_val<<" in line #"<<line_number<<". Line ignored.\n";
                success = false;
                continue;
            }

        /* Parse float properties */
            try{
                if (key =="living_reward" || key =="livingreward"){
                    this->setLivingReward(boost::lexical_cast<double>(val));
                    continue;
                }
                if (key == "deathpenalty" || key == "death_penalty"){
                    this->setDeathPenalty(boost::lexical_cast<double>(val));
                    continue;
                }
            }
            catch(boost::bad_lexical_cast &){
                std::cerr<<"WARNING! Loading config from: \""<<filename<<"\". Float value expected insted of: "<<raw_val<<" in line #"<<line_number<<". Line ignored.\n";
                success = false;
                continue;
            }
                        
        /* Parse string properties */
            if(key == "doom_map" || key == "doommap"){
                this->setDoomMap(val);
                continue;
            }
            if(key == "vizdoom_path" || key == "vizdoompath"){
                this->setViZDoomPath(original_val);
                continue;
            }
            if(key == "doom_game_path" || key == "doomgamepath"){
                this->setDoomGamePath(original_val);
                continue;
            }
            if(key == "doom_scenario_path" || key == "doomscenariopath"){
                this->setDoomScenarioPath(original_val);
                continue;
            }
            if(key == "doom_config_path" || key == "doomconfigpath"){
                this->setDoomConfigPath(original_val);
                continue;
            }

    
        /* Parse bool properties */
            try{
                if (key =="console_enabled" || key =="consoleenabled"){
                    this->setConsoleEnabled(StringToBool(val));
                    continue;
                }
                if (key =="sound_enabled" || key =="soundenabled"){
                    this->setSoundEnabled(StringToBool(val));
                    continue;
                }
                if (key =="render_hud" || key =="renderhud"){
                    this->setRenderHud(StringToBool(val));
                    continue;
                }
                if (key =="render_weapon" || key =="renderweapon"){
                    this->setRenderWeapon(StringToBool(val));
                    continue;
                }
                if (key =="render_crosshair" || key =="rendercrosshair"){
                    this->setRenderCrosshair(StringToBool(val));
                    continue;
                }
                if (key =="render_particles" || key =="renderparticles"){
                    this->setRenderParticles(StringToBool(val));
                    continue;
                }
                if (key =="render_decals" || key =="renderdecals"){
                    this->setRenderDecals(StringToBool(val));
                    continue;
                }
                if (key =="window_visible" || key =="windowvisible"){
                    this->setWindowVisible(StringToBool(val));
                    continue;
                }
               
            }
            catch( std::exception ){
                std::cerr<<"WARNING! Loading config from: \""<<filename<<"\". Boolean value expected insted of: "<<raw_val<<" in line #"<<line_number<<". Line ignored.\n";
                success = false;
                continue;
                            
            }

        /* Parse enum properties */

            if(key =="mode")
            {
                if(val == "spectator"){
                    this->setMode(SPECTATOR);
                    continue;
                }
                if(val == "player"){
                    this->setMode(PLAYER);
                    continue;
                }
                if(val == "spectator"){
                    this->setMode(SPECTATOR);
                    continue;
                }
                if(val == "async_player"){
                    this->setMode(ASYNC_PLAYER);
                    continue;
                }
                if(val == "async_spectator"){
                    this->setMode(ASYNC_SPECTATOR);
                    continue;
                }
                std::cerr<<"WARNING! Loading config from: \""<<filename<<"\". (ASYNC_)SPECTATOR || PLAYER expected instead of: "<<raw_val<<" in line #"<<line_number<<". Line ignored.\n";
                success = false;
                continue;  
            }

            try{
                if(key == "screen_resolution" || key == "screenresolution"){
                    this->setScreenResolution(StringToResolution(val));
                    continue;
                }
                if(key == "screen_format" || key == "screenformat"){
                    this->setScreenFormat(StringToFormat(val));
                    continue;
                }
                if(key == "button_max_value" || key == "buttonmaxvalue"){
                    size_t space = val.find_first_of(" ");
                    if(space == std::string::npos)
                        throw std::exception();
                    Button button = DoomGame::StringToButton(val.substr(0,space));
                    val = val.substr(space+1);
                    unsigned int max_value = boost::lexical_cast<unsigned int>(val);
                    if(val[0] == '-')
                        throw boost::bad_lexical_cast();
                    this->setButtonMaxValue(button,max_value);
                    continue;
                }
            }
            catch(std::exception&)
            {
                std::cerr<<"WARNING! Loading config from: \""<<filename<<"\". Unsupported value: "<<raw_val<<" in line #"<<line_number<<". Line ignored.\n";
                success = false;
                continue;
            }

            std::cerr<<"WARNING! Loading config from: \""<<filename<<"\". Unsupported key: "<<key<<" in line #"<<line_number<<". Line ignored.\n";
            success = false;
            
        }

        return success;
    }

}




