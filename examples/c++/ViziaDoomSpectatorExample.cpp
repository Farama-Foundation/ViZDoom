#include "ViziaDoomGame.h"
#include <iostream>
#include <vector>

using namespace Vizia;

int main(){

    DoomGame* dg = new DoomGame();

    std::cout << "\n\nVIZIA MAIN EXAMPLE\n\n";

    dg->setDoomGamePath("viziazdoom");
    dg->setDoomIwadPath("../scenarios/doom2.wad");
    dg->setDoomFilePath("../scenarios/s1_b.wad");
    dg->setDoomMap("map01");
    dg->setEpisodeTimeout(2000);
    dg->setEpisodeStartTime(1);
    dg->setGameMode(SPECTATOR);

    dg->setScreenResolution(640, 480);

    dg->setRenderHud(false);
    dg->setRenderCrosshair(false);
    dg->setRenderWeapon(true);
    dg->setRenderDecals(false);
    dg->setRenderParticles(false);

    dg->setVisibleWindow(true);

    dg->setDisabledConsole(true);

    dg->addAvailableButton(TURN_LEFT);
    dg->addAvailableButton(TURN_RIGHT);
    dg->addAvailableButton(MOVE_FORWARD);
    dg->addAvailableButton(MOVE_BACKWARD);
    dg->addAvailableButton(ATTACK);

    dg->addStateAvailableVar(HEALTH);
    dg->addStateAvailableVar(AMMO1);


    dg->init();
    //dg->newEpisode();
    std::vector<bool> action(3);

    action[0] = false;
    action[1] = false;
    action[2] = true;

    int iterations = 10000;
    int ep=1;
    for(int i = 0;i<iterations; ++i){

        if( dg->isEpisodeFinished() ){
            //std::cout << ep++ << std::endl;
            dg->newEpisode();
            // usleep(2000000);
        }
        DoomGame::State s = dg->getState();
        std::cout << "STATE NUMBER: " << s.number << " HP: " << s.vars[0] << " AMMO: " << s.vars[1] << std::endl;
        dg->advanceAction();
        //float r = dg->makeAction(action);
        //std::cout<<"reward: "<<r<<std::endl;
        //usleep(11000);
    }
    dg->close();
    delete dg;
}
