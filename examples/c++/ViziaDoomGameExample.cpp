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

    v->addAvailableButton(MOVE_LEFT);
    v->addAvailableButton(MOVE_RIGHT);
    v->addAvailableButton(ATTACK);

    v->addStateAvailableVar(HEALTH);
    v->addStateAvailableVar(AMMO1);


    v->init();
    //v->newEpisode();
    std::vector<int> action(3);

    action[0] = 0;
    action[1] = 0;
    action[2] = 1;

    int iterations = 10000;
    int ep=1;
    for(int i = 0;i<iterations; ++i){

        if( v->isEpisodeFinished() ){
            //std::cout << ep++ << std::endl;
            v->newEpisode();
           // usleep(2000000);
        }
        DoomGame::State s = v->getState();

        std::cout << "STATE NUMBER: " << s.number << " HP: " << s.vars[0] << " AMMO: " << s.vars[1] << std::endl;

        v->setNextAction(action);
        v->advanceAction(true, true, 4);

        std::cout<<"reward: "<<v->getLastReward()<<std::endl;
        //usleep(11000);
    }
    v->close();
    delete v;
}
