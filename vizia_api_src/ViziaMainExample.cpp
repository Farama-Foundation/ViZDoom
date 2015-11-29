#include "ViziaMain.h"
#include <iostream>
#include <unistd.h>
#include <vector>

int main(){

    ViziaMain* v= new ViziaMain();

    std::cout << "\n\nVIZIA MAIN EXAMPLE\n\n";

    v->setDoomGamePath("zdoom");
    v->setDoomIwadPath("doom2.wad");
    v->setDoomFilePath("s1_b.wad");
    v->setDoomMap("map01");
    v->setEpisodeTimeoutInDoomTics(200);

    v->setScreenResolution(320, 240);

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

    std::vector<bool> action(3);

    action[0] = false;
    action[1] = false;
    action[2] = true;

    int iterations = 1000;
    for(int i = 0; i < iterations; ++i){

        if( v->isEpisodeFinished() ){
            std::cout << "\nNEW EPISODE\n\n";
            v->newEpisode();
            usleep(2000000);
        }
        //ViziaMain::State s = v->getState();

        //std::cout << "STATE NUMBER: " << s.number <<
        //" HP: " << s.vars[0] << " AMMO: " << s.vars[1] << std::endl;

        float r = v->makeAction(action);
        std::cout<<"reward: "<<r<<std::endl;
        usleep(11000);
    }

    v->close();
    delete v;
}