#include "ViziaDoomGame.h"

#include <boost/lexical_cast.hpp>
#include <cmath> //  floor, ceil
#include <iostream> // cerr, cout
#include <vector>
#include <algorithm>
#include <fstream> // ifstream
#include <string> // getline
#include <cstdlib> // atoi
#include <stdexcept> // invalid_argument
#include <boost/algorithm/string.hpp> // to_lower
#include <boost/algorithm/string/trim_all.hpp> // trim_all
#include <exception>

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
        this->mode = PLAYER;

        this->doomController = new DoomController();
    }

    DoomGame::~DoomGame() {
        this->close();
        delete this->doomController;
    }

    bool DoomGame::ParseBool(std::string boolString)
    {
        if(boolString == "true" or boolString == "1")
            return true;
        if(boolString == "false" or boolString == "0")
            return false;
        throw std::exception();
    }

    ScreenResolution DoomGame::StringToResolution(std::string str)
    {
        if(str == "res_40x30")
            return RES_40X30;
        if(str == "res_60x45")
            return RES_60X45;
        if(str == "res_80x50")
            return RES_80X50;
        if(str == "res_80x60")
            return RES_80X60;
        if(str == "res_100x75")
            return RES_100X75;
        if(str == "res_120x75")
            return RES_120X75;
        if(str == "res_120x90")
            return RES_120X90;
        if(str == "res_160x100")
            return RES_160X100;
        if(str == "res_160x120")
            return RES_160X120;
        if(str == "res_200x120")
            return RES_200X120;
        if(str == "res_200x150")
            return RES_200X150;
        if(str == "res_240x135")
            return RES_240X135;
        if(str == "res_240x150")
            return RES_240X150;
        if(str == "res_240x180")
            return RES_240X180;
        if(str == "res_256x144")
            return RES_256X144;
        if(str == "res_256x160")
            return RES_256X160;
        if(str == "res_256x192")
            return RES_256X192;
        if(str == "res_320x200")
            return RES_320X200;
        if(str == "res_320x240")
            return RES_320X240;
        if(str == "res_400x225")
            return RES_400X225;
        if(str == "res_400x300")
            return RES_400X300;
        if(str == "res_480x270")
            return RES_480X270;
        if(str == "res_480x360")
            return RES_480X360;
        if(str == "res_512x288")
            return RES_512X288;
        if(str == "res_512x384")
            return RES_512X384;
        if(str == "res_640x360")
            return RES_640X400;
        if(str == "res_640x400")
            return RES_640X400;
        if(str == "res_640x480")
            return RES_640X480;
        if(str == "res_720x480")
            return RES_720X480;
        if(str == "res_720x540")
            return RES_720X540;
        if(str == "res_800x450")
            return RES_800X450;
        if(str == "res_800x480")
            return RES_800X480;
        if(str == "res_800x500")
            return RES_800X500;
        if(str == "res_800x600")
            return RES_800X600;
        if(str == "res_848x480")
            return RES_848X480;
        if(str == "res_960x600")
            return RES_960X600;
        if(str == "res_960x720")
            return RES_960X720;
        if(str == "res_1024x576")
            return RES_1024X576;
        if(str == "res_1024x600")
            return RES_1024X600;
        if(str == "res_1024x640")
            return RES_1024X640;
        if(str == "res_1024x768")
            return RES_1024X768;
        if(str == "res_1088x612")
            return RES_1088X612;
        if(str == "res_1152x648")
            return RES_1152X648;
        if(str == "res_1152x720")
            return RES_1152X720;
        if(str == "res_1152x864")
            return RES_1152X864;
        if(str == "res_1280x720")
            return RES_1280X720;
        if(str == "res_1280x854")
            return RES_1280X854;
        if(str == "res_1280x800")
            return RES_1280X800;
        if(str == "res_1280x960")
            return RES_1280X960;
        if(str == "res_1280x1024")
            return RES_1280X1024;
        if(str == "res_1360x768")
            return RES_1360X768;
        if(str == "res_1366x768")
            return RES_1366X768;
        if(str == "res_1400x787")
            return RES_1400X787;
        if(str == "res_1400x875")
            return RES_1400X875;
        if(str == "res_1400x1050")
            return RES_1400X1050;
        if(str == "res_1440x900")
            return RES_1440X900;
        if(str == "res_1440x960")
            return RES_1440X960;
        if(str == "res_1440x1080")
            return RES_1440X1080;
        if(str == "res_1600x900")
            return RES_1600X900;
        if(str == "res_1600x1000")
            return RES_1600X1000;
        if(str == "res_1600x1200")
            return RES_1600X1200;
        if(str == "res_1680x1050")
            return RES_1680X1050;
        if(str == "res_1920x1080")
            return RES_1920X1080;
        if(str == "res_1920x1200")
            return RES_1920X1200;
        if(str == "res_2048x1536")
            return RES_2048X1536;
        if(str == "res_2560x1440")
            return RES_2560X1440;
        if(str == "res_2560x1600")
            return RES_2560X1600;
        if(str == "res_2560x2048")
            return RES_2560X2048;
        if(str == "res_2880x1800")
            return RES_2880X1800;
        if(str == "res_3200x1800")
            return RES_3200X1800;
        if(str == "res_3840x2160")
            return RES_3840X2160;
        if(str == "res_3840x2400")
            return RES_3840X2400;
        if(str == "res_4096x2160")
            return RES_4096X2160;
        if(str == "res_5120x2880")
            return RES_5120X2880;
        throw std::exception();
    }
        
    ScreenFormat DoomGame::StringToFormat(std::string str)
    {
        if(str == "crcgcb")
            return CRCGCB;
        if(str == "crcgcbzb")
            return CRCGCBZB;
        if(str == "rgb24")
            return RGB24;
        if(str == "rgba32")
            return RGBA32;
        if(str == "argb32")
            return ARGB32;
        if(str == "cbcgcr")
            return CBCGCR;
        if(str == "cbcgcrzb")
            return CBCGCRZB;
        if(str == "bgr24")
            return BGR24;
        if(str == "bgra32")
            return BGRA32;
        if(str == "abgr32")
            return ABGR32;
        if(str == "gray8")
            return GRAY8;
        if(str == "zbuffer8")
            return ZBUFFER8;
        if(str == "doom_256_colors")
            return DOOM_256_COLORS;
        throw std::exception();
    }

    Button DoomGame::StringToButton(std::string)
    {
        //TODO
        throw std::exception();
    }
    GameVariable DoomGame::StringToGameVariable(std::string)
    {
        //TODO
        throw std::exception();
    }

    bool DoomGame::loadConfig(std::string filename) {
        bool success = true;
        std::ifstream file(filename.c_str());
        
        if(!file.good() )
        {
            std::cerr<<"WARNING! Loading config from: \""<<filename<<"\" failed. Something's wrong with the file. Check your spelling and permissions.\n";
            return false;
        }
        std::string line;
        int line_number = 0;

        /* Read every line. */
        while(!file.eof())
        {
            ++line_number;
            using namespace boost::algorithm;

            std::getline(file, line);

            /* Ignore empty and comment lines */
            trim_all(line);

            if(line.empty() or line[0] == '#'){
                continue;
            }

            /* Check if = is there */
            int equals_sign_pos = line.find_first_of('=');
            std::string key;
            std::string val;
            std::string raw_val;
            if( equals_sign_pos != std::string::npos )
            {
                key = line.substr(0,equals_sign_pos);
                val = line.substr(equals_sign_pos + 1);
                raw_val = val;
                trim_all(key);
                trim_all(val);
                to_lower(val);
                to_lower(key);
            }
            else
            {
                std::cerr<<"WARNING! Loading config from: \""<<filename<<"\". Missing \"=\" in line #"<<line_number<<". Line ignored.\n";
                success = false;
                continue;
            }
            
            if(key.empty())
            {
                std::cerr<<"WARNING! Loading config from: \""<<filename<<"\". Empty key in line #"<<line_number<<". Line ignored.\n";
                success = false;
                continue;
            }
            if(val.empty())
            {
                std::cerr<<"WARNING! Loading config from: \""<<filename<<"\". Empty value in line #"<<line_number<<". Line ignored.\n";
                success = false;
                continue;
            }
            /* Parse int properties */
            try{
                if (key =="seed"){
                    unsigned int value = boost::lexical_cast<unsigned int>(val);
                    if(val[0] == '-')
                        throw boost::bad_lexical_cast();
                    this->setSeed((unsigned int)value);
                    continue;
                }
                if (key == "episodetimeout" or key == "episode_timeout"){
                    unsigned int value = boost::lexical_cast<unsigned int>(val);
                    if(val[0] == '-')
                        throw boost::bad_lexical_cast();
                    this->setEpisodeTimeout((unsigned int)value);
                    continue;
                }
                if (key == "doom_skill" or key == "doomskill"){
                    unsigned int value = boost::lexical_cast<unsigned int>(val);
                    if(val[0] == '-')
                        throw boost::bad_lexical_cast();
                    this->setDoomSkill((unsigned int)value);
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
                if (key =="living_reward" or key =="livingreward"){
                    float value = boost::lexical_cast<float>(val);
                    this->setLivingReward((unsigned int)value);
                    continue;
                }
                if (key == "deathpenalty" or key == "death_penalty"){
                    float value = boost::lexical_cast<float>(val);
                    this->setDeathPenalty((unsigned int)value);
                    continue;
                }

            }
            catch(boost::bad_lexical_cast &){
                std::cerr<<"WARNING! Loading config from: \""<<filename<<"\". Float value expected insted of: "<<raw_val<<" in line #"<<line_number<<". Line ignored.\n";
                success = false;
                continue;
            }
            
            /* Parse string properties */
            if(key == "doom_map" or key == "doommap"){
                this->setDoomMap(val);
                continue;
            }
            if(key == "doom_game_path" or key == "doomgamepath"){
                this->setDoomGamePath(val);
                continue;
            }
            if(key == "doom_iwad_path" or key == "doomiwadpath"){
                this->setDoomIwadPath(val);
                continue;
            }
            if(key == "doom_file_path" or key == "doomfilepath"){
                this->setDoomFilePath(val);
                continue;
            }
            if(key == "doom_config_path" or key == "doomconfigpath"){
                this->setDoomConfigPath(val);
                continue;
            }
    
            /* Parse bool properties */
            try{
                if (key =="auto_new_episode" or key =="autonewepisode"){
                    this->setAutoNewEpisode(ParseBool(val));
                    continue;
                }
                if (key =="new_episode_on_timeout" or key =="newepisodeontimeout"){
                    this->setNewEpisodeOnTimeout(ParseBool(val));
                    continue;
                }
                if (key =="new_episode_on_player_death" or key =="newepisodeonplayerdeath"){
                    this->setNewEpisodeOnPlayerDeath(ParseBool(val));
                    continue;
                }
                if (key =="new_episode_on_map_end" or key =="newepisodeonmapend"){
                    this->setNewEpisodeOnMapEnd(ParseBool(val));
                    continue;
                }
                if (key =="console_enabled" or key =="consoleenabled"){
                    this->setConsoleEnabled(ParseBool(val));
                    continue;
                }
                if (key =="render_hud" or key =="renderhud"){
                    this->setRenderHud(ParseBool(val));
                    continue;
                }
                if (key =="render_weapon" or key =="renderweapon"){
                    this->setRenderWeapon(ParseBool(val));
                    continue;
                }
                if (key =="render_corsshair" or key =="rendercorsshair"){
                    this->setRenderCrosshair(ParseBool(val));
                    continue;
                }
                if (key =="render_particles" or key =="renderparticles"){
                    this->setRenderDecals(ParseBool(val));
                    continue;
                }
                if (key =="window_visible" or key =="windowvisible"){
                    this->setWindowVisible(ParseBool(val));
                    continue;
                }
               
            }
            catch( std::exception )
            {
                std::cerr<<"WARNING! Loading config from: \""<<filename<<"\". Boolean value expected insted of: "<<raw_val<<" in line #"<<line_number<<". Line ignored.\n";
                continue;
                success = false;            
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
                
                std::cerr<<"WARNING! Loading config from: \""<<filename<<"\". SPECTATOR or PLAYER expected instead of: "<<raw_val<<" in line #"<<line_number<<". Line ignored.\n";
                success = false;
                continue;
                
            }

            try{
                if(key == "screen_resolution" or key == "screenresolution"){
                    this->setScreenResolution(StringToResolution(val));
                    continue;
                }
                if(key == "screen_format" or key == "screenformat"){
                    this->setScreenFormat(StringToFormat(val));
                    continue;
                }

            }
            catch(std::exception)
            {
                std::cerr<<"WARNING! Loading config from: \""<<filename<<"\". Unsupported value: "<<raw_val<<" in line #"<<line_number<<". Line ignored.\n";
                success = false;
                continue;
            }
            //std::cerr<<"WARNING! Loading config from: \""<<filename<<"\". Unsupported key: "<<key<<" in line #"<<line_number<<". Line ignored.\n";
            success = false;
            
        }

        return success;
    }

    bool DoomGame::saveConfig(std::string filename) {
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

            if(this->mode == PLAYER){
                this->doomController->setAllowDoomInput(false);
            }
            else if(this->mode == SPECTATOR){
                this->doomController->setAllowDoomInput(true);
            }

            try {
                this->running = this->doomController->init();

                this->doomController->disableAllButtons();
                for (int i = 0; i < this->availableButtons.size(); ++i) {
                    this->doomController->setButtonAvailable(this->availableButtons[i], true);
                }

                this->state.gameVariables.resize(this->availableGameVariables.size());

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
        this->state.gameVariables.clear();
        this->lastAction.clear();

        this->running = false;
    }

    bool DoomGame::isRunning(){
        return this->running && this->doomController->isDoomRunning();
    }

    void DoomGame::newEpisode() {

        if(!this->isRunning()) throw DoomIsNotRunningException();

        this->doomController->restartMap();

        this->updateState();

        this->lastReward = 0.0;
        this->lastMapReward = 0.0;
        this->summaryReward = 0.0;
    }

    void DoomGame::setAction(std::vector<int> &actions) {

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
        this->advanceAction(true, true, 1);
    }

    void DoomGame::advanceAction(bool updateState, bool renderOnly, unsigned int tics) {

        if (!this->isRunning()) throw DoomIsNotRunningException();

        try {
            if(this->mode == PLAYER) this->doomController->tics(tics, updateState || renderOnly);
            else if(this->mode == SPECTATOR) this->doomController->realTimeTics(tics, updateState || renderOnly);
        }
        catch(const Exception &e){ throw; }

        if(updateState) this->updateState();
    }

    float DoomGame::makeAction(std::vector<int> &actions){
        this->setAction(actions);
        this->advanceAction();
        return this->getLastReward();
    }

    float DoomGame::makeAction(std::vector<int> &actions, unsigned int tics){
        this->setAction(actions);
        this->advanceAction(true, true, tics);
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
            for (int i = 0; i < this->availableGameVariables.size(); ++i) {
                this->state.gameVariables[i] = this->doomController->getGameVariable(this->availableGameVariables[i]);
            }

            /* Update float rgb image */
            this->state.number = this->doomController->getMapTic();
            this->state.imageBuffer = this->doomController->getScreen();

            this->lastStateNumber = this->state.number;

            if (this->mode == SPECTATOR) {
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

    Mode DoomGame::getMode(){ return this->mode; };
    void DoomGame::setMode(Mode mode){ if (!this->running) this->mode = mode; }

    const DoomController* DoomGame::getController() { return this->doomController; }

    int DoomGame::getGameVariable(GameVariable var){
        if(!this->isRunning()) throw DoomIsNotRunningException();

        return this->doomController->getGameVariable(var);
    }

    void DoomGame::setDoomGamePath(std::string path) { this->doomController->setGamePath(path); }
    void DoomGame::setDoomIwadPath(std::string path) { this->doomController->setIwadPath(path); }
    void DoomGame::setDoomFilePath(std::string path) { this->doomController->setFilePath(path); }
    void DoomGame::setDoomMap(std::string map) { this->doomController->setMap(map); }
    void DoomGame::setDoomSkill(int skill) { 
        //TODO warning when out of range
        this->doomController->setSkill(skill); 
    }
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

    void DoomGame::setScreenResolution(ScreenResolution resolution) {
        unsigned int width = 0, height = 0;

#define CASE_RES(w, h) case RES_##w##X##h : width = w; height = h; break;
        switch(resolution){
            CASE_RES(40, 30)
            CASE_RES(60, 45)
            CASE_RES(80, 50)
            CASE_RES(80, 60)
            CASE_RES(100, 75)
            CASE_RES(120, 75)
            CASE_RES(120, 90)
            CASE_RES(160, 100)
            CASE_RES(160, 120)
            CASE_RES(200, 120)
            CASE_RES(200, 150)
            CASE_RES(240, 135)
            CASE_RES(240, 150)
            CASE_RES(240, 180)
            CASE_RES(256, 144)
            CASE_RES(256, 160)
            CASE_RES(256, 192)
            CASE_RES(320, 200)
            CASE_RES(320, 240)
            CASE_RES(400, 225)	// 16:9
            CASE_RES(400, 300)
            CASE_RES(480, 270)	// 16:9
            CASE_RES(480, 360)
            CASE_RES(512, 288)	// 16:9
            CASE_RES(512, 384)
            CASE_RES(640, 360)	// 16:9
            CASE_RES(640, 400)
            CASE_RES(640, 480)
            CASE_RES(720, 480)	// 16:10
            CASE_RES(720, 540)
            CASE_RES(800, 450)	// 16:9
            CASE_RES(800, 480)
            CASE_RES(800, 500)	// 16:10
            CASE_RES(800, 600)
            CASE_RES(848, 480)	// 16:9
            CASE_RES(960, 600)	// 16:10
            CASE_RES(960, 720)
            CASE_RES(1024, 576)	// 16:9
            CASE_RES(1024, 600)	// 17:10
            CASE_RES(1024, 640)	// 16:10
            CASE_RES(1024, 768)
            CASE_RES(1088, 612)	// 16:9
            CASE_RES(1152, 648)	// 16:9
            CASE_RES(1152, 720)	// 16:10
            CASE_RES(1152, 864)
            CASE_RES(1280, 720)	// 16:9
            CASE_RES(1280, 854)
            CASE_RES(1280, 800)	// 16:10
            CASE_RES(1280, 960)
            CASE_RES(1280, 1024)	// 5:4
            CASE_RES(1360, 768)	// 16:9
            CASE_RES(1366, 768)
            CASE_RES(1400, 787)	// 16:9
            CASE_RES(1400, 875)	// 16:10
            CASE_RES(1400, 1050)
            CASE_RES(1440, 900)
            CASE_RES(1440, 960)
            CASE_RES(1440, 1080)
            CASE_RES(1600, 900)	// 16:9
            CASE_RES(1600, 1000)	// 16:10
            CASE_RES(1600, 1200)
            CASE_RES(1680, 1050)	// 16:10
            CASE_RES(1920, 1080)
            CASE_RES(1920, 1200)
            CASE_RES(2048, 1536)
            CASE_RES(2560, 1440)
            CASE_RES(2560, 1600)
            CASE_RES(2560, 2048)
            CASE_RES(2880, 1800)
            CASE_RES(3200, 1800)
            CASE_RES(3840, 2160)
            CASE_RES(3840, 2400)
            CASE_RES(4096, 2160)
            CASE_RES(5120, 2880)
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
    void DoomGame::setConsoleEnabled(bool console) {
        this->doomController->setNoConsole(!console);
    }
    int DoomGame::getScreenWidth() { return this->doomController->getScreenWidth(); }
    int DoomGame::getScreenHeight() { return this->doomController->getScreenHeight(); }
    int DoomGame::getScreenChannels() { return this->doomController->getScreenChannels(); }
    size_t DoomGame::getScreenPitch() { return this->doomController->getScreenPitch(); }
    size_t DoomGame::getScreenSize() { return this->doomController->getScreenSize(); }
    ScreenFormat DoomGame::getScreenFormat() { return this->doomController->getScreenFormat(); }

}
