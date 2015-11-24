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
            float* image;
        };
        struct StateFormat
        {
            //TODO some way to disable changing var_len and image_shape
            /* shape[0] - num. of channels, 1 - y, 2 - x */
            int image_shape[3];
            int var_len;
            StateFormat(int channels, int y, int x, int var_len)
            {
                this->image_shape[0] = channels;
                this->image_shape[1] = y;
                this->image_shape[2] = x;
                this->var_len = var_len;
            }
        };
        ViziaMain();
        ~ViziaMain();

        bool loadConfig(std::string file);
        bool saveConfig(std::string file);

        void init();
        void close();

        void newEpisode();
        float makeAction(std::vector<bool>& actions);

        ViziaMain::State getState();
        bool * getLastActions();
        bool isNewEpisode();
        bool isEpisodeFinished();

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

        void setAutoNewEpisode(bool set);
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

        StateFormat getStateFormat();
        int getActionFormat();
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

        bool initialized;

};

#endif
