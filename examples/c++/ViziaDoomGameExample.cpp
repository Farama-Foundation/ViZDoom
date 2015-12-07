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
    v->setEpisodeTimeoutInDoomTics(200);

    v->setScreenResolution(100, 0);

    v->setRenderHud(false);
    v->setRenderCrosshair(false);
    v->setRenderWeapon(true);
    v->setRenderDecals(false);
    v->setRenderParticles(false);

    v->addAvailableButton(MOVELEFT);
    v->addAvailableButton(MOVERIGHT);
    v->addAvailableButton(ATTACK);

    //v->addStateAvailableVar(HEALTH);
    v->addStateAvailableVar(AMMO1);


    v->init();
    //v->newEpisode();
    std::vector<bool> action(3);

    action[0] = false;
    action[1] = false;
    action[2] = true;

    int iterations = 10000;
    int ep=1;
    for(int i = 0;i<iterations; ++i){

        if( v->isEpisodeFinished() ){
            //std::cout << ep++ << std::endl;
            v->newEpisode();
           // usleep(2000000);
        }
        //ViziaMain::State s = v->getState();

        //std::cout << "STATE NUMBER: " << s.number <<
        //" HP: " << s.vars[0] << " AMMO: " << s.vars[1] << std::endl;

        float r = v->makeAction(action);
        //std::cout<<"reward: "<<r<<std::endl;
        //usleep(11000);
    }

    v->close();
    delete v;
}
