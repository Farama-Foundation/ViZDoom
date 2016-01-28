#include "ViziaDoomGame.h"
#include <iostream>
#include <vector>
#include <unistd.h>

using namespace Vizia;

int main(){

    DoomGame* dg = new DoomGame();

    std::cout << "\n\nVIZIA MAIN EXAMPLE\n\n";

    dg->setDoomGamePath("viziazdoom");
    dg->setDoomIwadPath("../scenarios/doom2.wad");
    //dg->setDoomFilePath("../scenarios/s1_b.wad");
    dg->setDoomMap("map01");
    dg->setEpisodeTimeout(200);
    dg->setEpisodeStartTime(1);
    dg->setMode(SPECTATOR);

    dg->setScreenResolution(RES_640X480);

    dg->setRenderHud(false);
    dg->setRenderCrosshair(false);
    dg->setRenderWeapon(true);
    dg->setRenderDecals(false);
    dg->setRenderParticles(false);

    dg->setWindowVisible(true);

    dg->addAvailableButton(TURN_LEFT_RIGHT_DELTA, 30);
    dg->addAvailableButton(MOVE_FORWARD_BACKWARD_DELTA, 30);
    dg->addAvailableButton(MOVE_FORWARD);
    dg->addAvailableButton(TURN_LEFT);
    dg->addAvailableButton(TURN_RIGHT);
    dg->addAvailableButton(ATTACK);
    dg->addAvailableButton(SELECT_WEAPON1);
    dg->addAvailableButton(SELECT_WEAPON2);

    dg->addAvailableGameVariable(HEALTH);
    dg->addAvailableGameVariable(AMMO2);

    dg->init();

    int iterations = 10000;
    int ep=1;
    for(int i = 0;i<iterations; ++i){

        if( dg->isEpisodeFinished() ){
            //std::cout << ep++ << std::endl;
            dg->newEpisode();
            // usleep(2000000);
        }
        DoomGame::State s = dg->getState();
        std::cout << "STATE NUMBER: " << s.number << " HP: " << s.gameVariables[0] << " AMMO2: " << s.gameVariables[1] << std::endl;
        std::cout<<"TIC: " << dg->getEpisodeTime() << " LAST ACTION: " << dg->getLastAction()[0] << " " << dg->getLastAction()[1]
        << " " << dg->getLastAction()[2] << " " << dg->getLastAction()[3] << " " << dg->getLastAction()[4] << " " << dg->getLastAction()[5] << std::endl;
        dg->advanceAction();
    }
    dg->close();
    delete dg;
}
