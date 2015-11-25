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
        struct StateFormat{
            /* shape[0] - num. of channels, 1 - x, 2 - y */
            int imageShape[3];
            int varLen;
            StateFormat(){}
            StateFormat(int channels, int x, int y, int varLen)
            {
                this->imageShape[0] = channels;
                this->imageShape[1] = x;
                this->imageShape[2] = y;
                this->varLen = varLen;
            }
        };
        ViziaMain();
        ~ViziaMain();

        bool loadConfig(std::string file);
        bool saveConfig(std::string file);

        int init();
        void close();

        void newEpisode();
        float makeAction(std::vector<bool>& actions);

        ViziaMain::State getState();
        bool * getlastAction();
        bool isNewEpisode();
        bool isEpisodeFinished();

        void addAvailableKey(int action);
        void addAvailableKey(std::string action);

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
        std::vector<int> availableKeys;

        bool *lastAction;

        bool initialized;

        StateFormat stateFormat;
        State state;

};

#endif
