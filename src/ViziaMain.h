#ifndef __VIZIA_MAIN_H__
#define __VIZIA_MAIN_H__

#include <string>

#include "ViziaDoomController.h"

float DoomTic2S (unsigned int);

int DoomTic2Ms (unsigned int);

unsigned int S2DoomTic (float);

unsigned int Ms2DoomTic (float);

class ViziaMain{

    public:

        ViziaMain();
        ~ViziaMain();

        void loadConfig(std::string file);

        ViziaDoomController* getController();

        newEpisode();

        makeAction();

        getState();

        isEpisodeFinished();

    private:

        ViziaDoomController * doomController;

        std::vector<int> stateVars;
        std::vector<int> availableActions;

};


#endif
