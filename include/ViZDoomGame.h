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

#ifndef __VIZDOOM_GAME_H__
#define __VIZDOOM_GAME_H__

#include "ViZDoomDefines.h"
#include "ViZDoomController.h"

#include <string>
#include <vector>
#include <list>

namespace vizdoom {

    class DoomGame {

    public:

        DoomGame();
        virtual ~DoomGame();

        bool loadConfig(std::string filename);

        bool init();
        void close();

        void newEpisode();
        bool isRunning();

        void setAction(std::vector<int> const &actions);
        void advanceAction();
        void advanceAction(unsigned int tics);
        void advanceAction(unsigned int tics, bool updateState, bool renderOnly);

        double makeAction(std::vector<int> const &actions);
        double makeAction(std::vector<int> const &actions, unsigned int tics);
        
        GameState getState();

        std::vector<int> getLastAction();

        bool isNewEpisode();
        bool isEpisodeFinished();

        bool isPlayerDead();
        void respawnPlayer();

        void addAvailableButton(Button button);
        void addAvailableButton(Button button, int maxValue);
        void clearAvailableButtons();
        int getAvailableButtonsSize();
        void setButtonMaxValue(Button button, int maxValue);
        int getButtonMaxValue(Button button);

        void addAvailableGameVariable(GameVariable var);

        void clearAvailableGameVariables();
        int getAvailableGameVariablesSize();

        void addGameArgs(std::string args);
        void clearGameArgs();

        void sendGameCommand(std::string cmd);

        uint8_t * const getGameScreen();

        Mode getMode();
        void setMode(Mode mode);

        //OPTIONS

        int getGameVariable(GameVariable var);

        double getLivingReward();
        void setLivingReward(double livingReward);
        double getDeathPenalty();
        void setDeathPenalty(double deathPenalty);

        double getLastReward();
        double getTotalReward();

        void setViZDoomPath(std::string path);
        void setDoomGamePath(std::string path);
        void setDoomScenarioPath(std::string path);
        void setDoomMap(std::string map);
        void setDoomSkill(int skill);
        void setDoomConfigPath(std::string path);

        unsigned int getSeed();
        void setSeed(unsigned int seed);

        unsigned int getEpisodeStartTime();
        void setEpisodeStartTime(unsigned int tics);

        unsigned int getEpisodeTimeout();
        void setEpisodeTimeout(unsigned int tics);

        unsigned int getEpisodeTime();

        void setScreenResolution(ScreenResolution resolution);
        void setScreenFormat(ScreenFormat format);
        void setRenderHud(bool hud);
        void setRenderWeapon(bool weapon);
        void setRenderCrosshair(bool crosshair);
        void setRenderDecals(bool decals);
        void setRenderParticles(bool particles);
        void setWindowVisible(bool visibility);
        void setConsoleEnabled(bool console);
        void setSoundEnabled(bool sound);

        int getScreenWidth();
        int getScreenHeight();
        int getScreenChannels();
        size_t getScreenPitch();
        size_t getScreenSize();

        ScreenFormat getScreenFormat();

    protected:

        DoomController *doomController;
        bool running;

        /* STATE AND ACTIONS */
        Mode mode;

        GameState state;
        void updateState();

        std::vector <GameVariable> availableGameVariables;
        std::vector <Button> availableButtons;

        std::vector<int> lastAction;

        /* Reward */
        unsigned int nextStateNumber;
        unsigned int lastMapTic;

        double lastReward;
        double lastMapReward;
        double summaryReward;

        double livingReward;
        double deathPenalty;

    private:
        /* Load config helpers */
        static bool StringToBool(std::string boolString);
        static ScreenResolution StringToResolution(std::string str);
        static ScreenFormat StringToFormat(std::string str);
        static Button StringToButton(std::string str);
        static GameVariable StringToGameVariable(std::string str);
        static unsigned int StringToUint(std::string str);
        static bool ParseListProperty(int &line_number, std::string &value, std::ifstream& input, std::vector<std::string> &output);

    };
}

#endif
