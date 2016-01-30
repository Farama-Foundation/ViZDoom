#include "ViziaDoomGame.h"
#include <iostream>
#include <unistd.h>
#include <vector>

using namespace Vizia;

int main(){

    DoomGame* dg = new DoomGame();

    std::cout << "\n\nVIZIA MAIN EXAMPLE\n\n";

    dg->setDoomPath("viziazdoom");
    dg->setGameFilePath("../scenarios/doom2.wad");
    dg->setScenarioFilePath("../scenarios/s1_b.wad");
    dg->setDoomMap("map01");
    dg->setEpisodeTimeout(200);
    dg->setLivingReward(-1);

    dg->setScreenResolution(RES_640X480);

    dg->setRenderHud(false);
    dg->setRenderCrosshair(false);
    dg->setRenderWeapon(true);
    dg->setRenderDecals(false);
    dg->setRenderParticles(false);

    dg->setWindowVisible(true);

    dg->addAvailableButton(MOVE_LEFT_RIGHT_DELTA);
    dg->addAvailableButton(MOVE_FORWARD_BACKWARD_DELTA);
    dg->addAvailableButton(TURN_LEFT_RIGHT_DELTA);

    dg->init();
    //dg->newEpisode();
    std::vector<int> action(3);

    action[0] = -5;
    action[1] = 1;
    action[2] = -45;

    int iterations = 10000;
    int ep=1;
    for(int i = 0;i<iterations; ++i){

        if( dg->isEpisodeFinished() ){
            dg->newEpisode();
        }
        DoomGame::State s = dg->getState();

        std::cout << "STATE NUMBER: " << s.number << std::endl;

        dg->makeAction(action);
    }
    dg->close();
    delete dg;
}
