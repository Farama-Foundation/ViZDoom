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

#include "ViZDoomTypes.h"

#include <list>
#include <memory>
#include <string>
#include <vector>

namespace vizdoom {

    class DoomController;

    class DoomGame {

    public:

        DoomGame();
        virtual ~DoomGame();


        /* Flow Control */
        /*------------------------------------------------------------------------------------------------------------*/

        bool init();
        void close();
        void newEpisode(std::string filePath = "");
        void replayEpisode(std::string filePath, unsigned int player = 0);
        bool isRunning();

        void setAction(std::vector<int> const &actions);
        void advanceAction(unsigned int tics = 1, bool updateState = true);
        double makeAction(std::vector<int> const &actions, unsigned int tics = 1);

        bool isNewEpisode();
        bool isEpisodeFinished();
        bool isPlayerDead();
        void respawnPlayer();
        void sendGameCommand(std::string cmd);

        GameStatePtr getState();
        std::vector<int> getLastAction();


        /* Buttons settings */
        /*------------------------------------------------------------------------------------------------------------*/

        std::vector<Button> getAvailableButtons();
        void setAvailableButtons(std::vector<Button> buttons);

        void addAvailableButton(Button button, unsigned int maxValue = -1);
        void clearAvailableButtons();
        size_t getAvailableButtonsSize();

        void setButtonMaxValue(Button button, unsigned int maxValue);
        int getButtonMaxValue(Button button);


        /* GameVariables getters and setters */
        /*------------------------------------------------------------------------------------------------------------*/

        std::vector<GameVariable> getAvailableGameVariables();
        void setAvailableGameVariables(std::vector<GameVariable> gameVariables);

        void addAvailableGameVariable(GameVariable var);
        void clearAvailableGameVariables();
        size_t getAvailableGameVariablesSize();
        double getGameVariable(GameVariable variable);


        /* GameArgs getters and setters */
        /*------------------------------------------------------------------------------------------------------------*/

        void addGameArgs(std::string args);
        void clearGameArgs();


        /* Rewards getters and setters */
        /*------------------------------------------------------------------------------------------------------------*/

        double getLivingReward();
        void setLivingReward(double livingReward);
        double getDeathPenalty();
        void setDeathPenalty(double deathPenalty);
        double getLastReward();
        double getTotalReward();


        /* General game getters and setters */
        /*------------------------------------------------------------------------------------------------------------*/

        bool loadConfig(std::string filePath);
        Mode getMode();
        void setMode(Mode mode);

        unsigned int getTicrate();
        void setTicrate(unsigned int ticrate);

        void setViZDoomPath(std::string filePath);
        void setDoomGamePath(std::string filePath);
        void setDoomScenarioPath(std::string filePath);
        void setDoomMap(std::string map);
        void setDoomSkill(int skill);
        void setDoomConfigPath(std::string filePath);

        unsigned int getSeed();
        void setSeed(unsigned int seed);

        unsigned int getEpisodeStartTime();
        void setEpisodeStartTime(unsigned int tics);
        unsigned int getEpisodeTimeout();
        void setEpisodeTimeout(unsigned int tics);
        unsigned int getEpisodeTime();


        /* Output getters and setters */
        /*------------------------------------------------------------------------------------------------------------*/

        void setScreenResolution(ScreenResolution resolution);
        ScreenFormat getScreenFormat();
        void setScreenFormat(ScreenFormat format);

        bool isDepthBufferEnabled();
        void setDepthBufferEnabled(bool depthBuffer);

        bool isLabelsBufferEnabled();
        void setLabelsBufferEnabled(bool labelsBuffer);

        bool isAutomapBufferEnabled();
        void setAutomapBufferEnabled(bool automapBuffer);

        void setAutomapMode(AutomapMode mode);
        void setAutomapRotate(bool rotate);
        void setAutomapRenderTextures(bool textures);

        void setRenderHud(bool hud);
        void setRenderMinimalHud(bool minHud);
        void setRenderWeapon(bool weapon);
        void setRenderCrosshair(bool crosshair);
        void setRenderDecals(bool decals);
        void setRenderParticles(bool particles);
        void setRenderEffectsSprites(bool sprites);
        void setRenderMessages(bool messages);
        void setRenderCorpses(bool bodies);
        void setWindowVisible(bool visibility);
        void setConsoleEnabled(bool console);
        void setSoundEnabled(bool sound);

        int getScreenWidth();
        int getScreenHeight();
        int getScreenChannels();
        size_t getScreenDepth();
        size_t getScreenPitch();
        size_t getScreenSize();


    protected:

        DoomController *doomController;

        /* Game state and actions */
        /*------------------------------------------------------------------------------------------------------------*/

        bool running;

        Mode mode;

        GameStatePtr state;

        void resetState();
        void updateState();

        std::vector<GameVariable> availableGameVariables;
        std::vector<Button> availableButtons;
        std::vector<int> lastAction;

        unsigned int nextStateNumber;
        unsigned int lastMapTic;

        /* Rewards */
        /*------------------------------------------------------------------------------------------------------------*/

        double lastReward;
        double lastMapReward;
        double summaryReward;

        double livingReward;
        double deathPenalty;

    private:



    };
}

#endif
