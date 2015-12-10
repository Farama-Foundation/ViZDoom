#ifndef __VIZIA_MAIN_H__
#define __VIZIA_MAIN_H__

#include "ViziaDefines.h"
#include "ViziaDoomController.h"

#include <string>
#include <vector>

namespace Vizia {

    unsigned int DoomTics2Ms(unsigned int tics);

    unsigned int Ms2DoomTics(unsigned int ms);

    class DoomGame {

    public:

        struct State {
            int number;
            std::vector<int> vars;
            uint8_t *imageBuffer;
        };

        DoomGame();
        virtual ~DoomGame();

        bool loadConfig(std::string file);
        bool saveConfig(std::string file);

        bool init();
        void close();

        void newEpisode();
        bool isRunning();

        float makeAction(std::vector<bool> &actions);

        State getState();

        std::vector<bool> getLastAction();

        bool isNewEpisode();
        bool isEpisodeFinished();

        void addAvailableButton(Button button);
        //void addAvailableButton(std::string action);

        void addStateAvailableVar(GameVar var);
        //void addStateAvailableVar(std::string var);

        //OPTIONS

        const DoomController *getController();

        int getGameVar(GameVar var);

        int getLivingReward();
        void setLivingReward(int livingReward);
        int getDeathPenalty();
        void setDeathPenalty(int deathPenalty);

        int getLastReward();
        int getSummaryReward();

        void setDoomGamePath(std::string path);
        void setDoomIwadPath(std::string path);
        void setDoomFilePath(std::string path);
        void setDoomMap(std::string map);
        void setDoomSkill(int skill);
        void setDoomConfigPath(std::string path);

        void setAutoNewEpisode(bool set);
        void setNewEpisodeOnTimeout(bool set);
        void setNewEpisodeOnPlayerDeath(bool set);
        void setNewEpisodeOnMapEnd(bool set);

        //void setEpisodeStartTimeInMiliseconds(unsigned int ms);
        //void setEpisodeStartTimeInDoomTics(unsigned int tics);

        unsigned int getEpisodeTimeoutInMiliseconds();
        void setEpisodeTimeoutInMiliseconds(unsigned int ms);

        unsigned int getEpisodeTimeoutInDoomTics();
        void setEpisodeTimeoutInDoomTics(unsigned int tics);

        void setScreenResolution(unsigned int width, unsigned int height);
        void setScreenWidth(unsigned int width);
        void setScreenHeight(unsigned int height);
        void setScreenFormat(ScreenFormat format);
        void setRenderHud(bool hud);
        void setRenderWeapon(bool weapon);
        void setRenderCrosshair(bool crosshair);
        void setRenderDecals(bool decals);
        void setRenderParticles(bool particles);

        int getActionFormat();

        int getScreenWidth();
        int getScreenHeight();
        size_t getScreenPitch();
        size_t getScreenSize();

        ScreenFormat getScreenFormat();

    private:

        DoomController *doomController;
        bool running;

        //STATE AND ACTIONS
        State state;

        std::vector <GameVar> stateAvailableVars;
        std::vector <Button> availableButtons;

        std::vector<bool> lastAction;

        //REWARD

        int lastReward;
        int lastMapReward;
        int summaryReward;

        int livingReward;
        int deathPenalty;

    };
}

#endif
