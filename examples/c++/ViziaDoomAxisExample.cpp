#include "ViziaDoomGame.h"
#include <iostream>
#include <unistd.h>
#include <vector>

using namespace Vizia;

int main(){

    DoomGame* v= new DoomGame();

    std::cout << "\n\nVIZIA MAIN EXAMPLE\n\n";

    v->setDoomGamePath("viziazdoom");
    v->setDoomIwadPath("../scenarios/doom2.wad");
    v->setDoomFilePath("../scenarios/s1_b.wad");
    v->setDoomMap("map01");
    v->setEpisodeTimeout(200);
    v->setLivingReward(-1);

    v->setScreenResolution(640, 480);

    v->setRenderHud(false);
    v->setRenderCrosshair(false);
    v->setRenderWeapon(true);
    v->setRenderDecals(false);
    v->setRenderParticles(false);

    v->setVisibleWindow(true);

    v->setDisabledConsole(true);

    v->addAvailableButton(LEFT_RIGHT);
    v->addAvailableButton(FORWARD_BACKWARD);
    v->addAvailableButton(VIEW_ANGLE);

    v->init();
    //v->newEpisode();
    std::vector<int> action(3);

    action[0] = -5;
    action[1] = 1;
    action[2] = 100;

    int iterations = 10000;
    int ep=1;
    for(int i = 0;i<iterations; ++i){

        if( v->isEpisodeFinished() ){
            v->newEpisode();
        }
        DoomGame::State s = v->getState();

        std::cout << "STATE NUMBER: " << s.number << std::endl;

        v->setNextAction(action);
        v->advanceAction();
    }
    v->close();
    delete v;
}
