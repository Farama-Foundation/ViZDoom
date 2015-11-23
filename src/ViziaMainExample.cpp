#include "ViziaMain.h"
#include <iostream>
#include <unistd.h>
int main(){

    ViziaMain *v= new ViziaMain;

    std::cout << "\n\nVIZIA MAIN EXAMPLE\n\n";

    v->setDoomGamePath("zdoom");
    v->setDoomIwadPath("doom2.wad");
    v->setDoomFilePath("s1_b.wad");
    v->setDoomMap("map01");
    v->setEpisodeTimeoutInDoomTics(500);

    v->setScreenResolution(320, 240);

    v->setRenderHud(true);
    v->setRenderCrosshair(true);
    v->setRenderWeapon(true);
    v->setRenderDecals(false);
    v->setRenderParticles(false);

    v->addAvailableAction("MOVELEFT");
    v->addAvailableAction("MOVERIGHT");
    v->addAvailableAction("ATTACK");

    v->addStateAvailableVar("HEALTH");
    v->addStateAvailableVar("AMMO1");

    v->init();

    int loop = 100;
    int iterations = 2000;

    for(int i = 0; i < iterations; ++i){

        bool *actions = new bool[3];
        actions[0]=true;
        actions[1]=false;
        actions[2]=false;


        ViziaMain::State s = v->getState();

        std::cout << "STATE NUMBER: " << s.number <<
        " HP: " << s.vars[0] << " AMMO: " << s.vars[1] << std::endl;

        v->makeAction(actions);
        usleep(10000);
    }

    v->close();
}