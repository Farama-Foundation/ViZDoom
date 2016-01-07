#include "ViziaDoomGame.h"
#include <iostream>
#include <unistd.h>
#include <vector>

using namespace Vizia;

int main(){

    DoomGame* dg= new DoomGame();

    std::cout << "\n\nVIZIA MAIN EXAMPLE\n\n";

    dg->setDoomGamePath("viziazdoom");
    dg->setDoomIwadPath("../scenarios/doom2.wad");
    dg->setDoomFilePath("../scenarios/health_gathering.wad");
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

    dg->addAvailableButton(MOVE_LEFT);
    dg->addAvailableButton(MOVE_RIGHT);
    dg->addAvailableButton(ATTACK);

    dg->addStateAvailableVar(HEALTH);
    dg->addStateAvailableVar(AMMO1);


    dg->init();
    //dg->newEpisode();
    std::vector<int> action(3);

    action[0] = 0;
    action[1] = 0;
    action[2] = 1;

    int iterations = 10000;
    int ep=1;
    for(int i = 0;i<iterations; ++i){

        if( dg->isEpisodeFinished() ){
            dg->newEpisode();
        }
        DoomGame::State s = dg->getState();

        std::cout << "STATE NUMBER: " << s.number << " HP: " << s.vars[0] << " AMMO: " << s.vars[1] << std::endl;

        dg->makeAction(action, 4);

        std::cout<<"reward: "<<dg->getLastReward()<<std::endl;
    }
    dg->close();
    delete dg;
}
