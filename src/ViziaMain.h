#ifndef __VIZIA_MAIN_H__
#define __VIZIA_MAIN_H__

#include <string>
#include <vector>
#include "ViziaDoomController.h"

unsigned int DoomTics2Ms (unsigned int tics);
unsigned int Ms2DoomTics (unsigned int ms);

class ViziaMain{

    public:

        struct State{
            int number;
            int* vars;
            int imageWidth;
            int imageHeight;
            int imagePitch;
            uint8_t* imageBuffer;
        };

        ViziaMain();
        ~ViziaMain();

        void loadConfig(std::string file);

        void init();
        void close();

        void newEpisode();
        void makeAction(const bool * actions);

        ViziaMain::State getState();
        bool * getLastActions();
        bool isNewEpisode();

        void addAvailableAction(int action);
        void addAvailableAction(std::string action);

        void addStateAvailableVar(int var);
        void addStateAvailableVar(std::string var);

        //OPTIONS

        const ViziaDoomController* getController();

        void setDoomGamePath(std::string path);
        void setDoomIwadPath(std::string path);
        void setDoomFilePath(std::string path);
        void setDoomMap(std::string map);
        void setDoomSkill(int skill);
        void setDoomConfigPath(std::string path);

        void setNewEpisodeOnTimeout(bool set);
        void setNewEpisodeOnPlayerDeath(bool set);
        void setEpisodeTimeoutInMiliseconds(unsigned int ms);
        void setEpisodeTimeoutInDoomTics(unsigned int tics);

        void setScreenResolution(int width, int height);
        void setScreenWidth(int width);
        void setScreenHeight(int height);
        void setScreenFormat(int format);
        void setRenderHud(bool hud);
        void setRenderWeapon(bool weapon);
        void setRenderCrosshair(bool crosshair);
        void setRenderDecals(bool decals);
        void setRenderParticles(bool particles);

        int getScreenWidth();
        int getScreenHeight();
        int getScreenPitch();
        int getScreenSize();
        int getScreenFormat();

    private:

        ViziaDoomController * doomController;

        std::vector<int> stateAvailableVars;
        std::vector<int> availableActions;

        int *stateVars;
        bool *lastActions;

};

#endif
