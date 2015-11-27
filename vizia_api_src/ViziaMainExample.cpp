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

    v->setRenderHud(true);
    v->setRenderCrosshair(true);
    v->setRenderWeapon(true);
    v->setRenderDecals(true);
    v->setRenderParticles(true);

    v->addAvailableButton(MOVELEFT);
    v->addAvailableButton(MOVERIGHT);
    v->addAvailableButton(ATTACK);

    v->addStateAvailableVar(HEALTH);
    v->addStateAvailableVar(AMMO1);

    //v->setAutoNewEpisode(true); //enables episode auto reloading
    std::cout<<"Expecting segfault:"<<std::endl;
    v->close();
    delete v;
    return 0;
    v->init();

    int iterations = 10000;
    for(int i = 0; i < iterations; ++i){

        if( v->isEpisodeFinished() ){
            std::cout << "\nNEW EPISODE\n\n";
            v->newEpisode();
        }

        std::vector<bool> actions(3);

        actions[0] = false;
        
        actions[1] = false;
        
        actions[2] = true;
        

        //ViziaMain::State s = v->getState();

        //std::cout << "STATE NUMBER: " << s.number <<
        //" HP: " << s.vars[0] << " AMMO: " << s.vars[1] << std::endl;

        v->makeAction(actions);
        //usleep(10000);
    }

    v->close();
}