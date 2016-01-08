#ifndef __VIZIA_MAIN_H__
#define __VIZIA_MAIN_H__

#include "ViziaDefines.h"
#include "ViziaDoomController.h"

#include <string>
#include <vector>

namespace Vizia {

    unsigned int DoomTics2Ms(unsigned int tics);

    unsigned int Ms2DoomTics(unsigned int ms);

    float DoomFixedToFloat(int doomFixed);

    class DoomGame {

    public:

        struct State {
            unsigned int number;
            std::vector<int> vars;
            uint8_t * imageBuffer;
        };

        DoomGame();
        virtual ~DoomGame();

        bool loadConfig(std::string file);
        bool saveConfig(std::string file);

        bool init();
        void close();

        void newEpisode();
        bool isRunning();

        void setAction(std::vector<int> &actions);
        void advanceAction();
        void advanceAction(bool stateUpdate, bool renderOnly, unsigned int tics);

        float makeAction(std::vector<int> &actions);
        float makeAction(std::vector<int> &actions, unsigned int tics);
        
        State getState();

        std::vector<int> getLastAction();

        bool isNewEpisode();
        bool isEpisodeFinished();

        void addAvailableButton(Button button);
        void addAvailableButton(Button button, int maxValue);
        void clearAvailableButtons();
        int getAvailableButtonsSize();
        void setButtonMaxValue(Button button, int maxValue);

        void addStateAvailableVar(GameVar var);
        void clearStateAvailableVars();
        int getStateAvailableVarsSize();

        void addCustomGameArg(std::string arg);
        void clearCustomGameArgs();

        void sendGameCommand(std::string cmd);

        uint8_t * const getGameScreen();

        GameMode getGameMode();
        void setGameMode(GameMode mode);

        //OPTIONS

        const DoomController *getController();

        int getGameVar(GameVar var);

        float getLivingReward();
        void setLivingReward(float livingReward);
        float getDeathPenalty();
        void setDeathPenalty(float deathPenalty);

        float getLastReward();
        float getSummaryReward();

        void setDoomGamePath(std::string path);
        void setDoomIwadPath(std::string path);
        void setDoomFilePath(std::string path);
        void setDoomMap(std::string map);
        void setDoomSkill(int skill);
        void setDoomConfigPath(std::string path);

        unsigned int getSeed();
        void setSeed(unsigned int seed);

        void setAutoNewEpisode(bool set);
        void setNewEpisodeOnTimeout(bool set);
        void setNewEpisodeOnPlayerDeath(bool set);
        void setNewEpisodeOnMapEnd(bool set);

        unsigned int getEpisodeStartTime();
        void setEpisodeStartTime(unsigned int tics);

        unsigned int getEpisodeTimeout();
        void setEpisodeTimeout(unsigned int tics);

        void setScreenResolution(ScreenResolution resolution);
        void setScreenWidth(unsigned int width);
        void setScreenHeight(unsigned int height);
        void setScreenFormat(ScreenFormat format);
        void setRenderHud(bool hud);
        void setRenderWeapon(bool weapon);
        void setRenderCrosshair(bool crosshair);
        void setRenderDecals(bool decals);
        void setRenderParticles(bool particles);
        void setWindowVisible(bool visibility);
        void setConsoleEnabled(bool console);

        int getScreenWidth();
        int getScreenHeight();
        int getScreenChannels();
        size_t getScreenPitch();
        size_t getScreenSize();

        ScreenFormat getScreenFormat();

    protected:

        DoomController *doomController;
        bool running;

        //STATE AND ACTIONS
        GameMode gameMode;

        State state;
        void updateState();

        std::vector <GameVar> stateAvailableVars;
        std::vector <Button> availableButtons;

        std::vector<int> lastAction;

        //REWARD
        unsigned int lastStateNumber;

        float lastReward;
        float lastMapReward;
        float summaryReward;

        float livingReward;
        float deathPenalty;

    };
}

#endif
