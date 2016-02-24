#include "ViziaDoomGame.h"
#include <iostream>
#include <vector>
#include <unistd.h>

using namespace Vizia;

int main(){

    DoomGame* game = new DoomGame();

    std::cout << "\n\nVIZIA MAIN EXAMPLE\n\n";

    game->setDoomEnginePath("viziazdoom");
    game->setDoomGamePath("../scenarios/doom2.wad");
    //game->setDoomScenarioPath("../scenarios/s1_b.wad");
    game->setDoomMap("map01");
    game->setEpisodeTimeout(200);
    game->setEpisodeStartTime(1);
    game->setMode(SPECTATOR);

    game->setScreenResolution(RES_640X480);

    game->setRenderHud(false);
    game->setRenderCrosshair(false);
    game->setRenderWeapon(true);
    game->setRenderDecals(false);
    game->setRenderParticles(false);

    game->setWindowVisible(true);

    game->addAvailableButton(TURN_LEFT_RIGHT_DELTA, 30);
    game->addAvailableButton(MOVE_FORWARD_BACKWARD_DELTA, 30);
    game->addAvailableButton(MOVE_FORWARD);
    game->addAvailableButton(TURN_LEFT);
    game->addAvailableButton(TURN_RIGHT);
    game->addAvailableButton(ATTACK);
    game->addAvailableButton(SELECT_WEAPON1);
    game->addAvailableButton(SELECT_WEAPON2);

    game->addAvailableGameVariable(HEALTH);
    game->addAvailableGameVariable(AMMO2);

    game->init();

    int iterations = 10000;
    int ep=1;
    for(int i = 0;i<iterations; ++i){

        if( game->isEpisodeFinished() ){
            //std::cout << ep++ << std::endl;
            game->newEpisode();
            // usleep(2000000);
        }
        DoomGame::State s = game->getState();
        std::cout << "STATE NUMBER: " << s.number << " HP: " << s.gameVariables[0] << " AMMO2: " << s.gameVariables[1] << std::endl;
        std::cout<<"TIC: " << game->getEpisodeTime() << " LAST ACTION: " << game->getLastAction()[0] << " " << game->getLastAction()[1]
        << " " << game->getLastAction()[2] << " " << game->getLastAction()[3] << " " << game->getLastAction()[4] << " " << game->getLastAction()[5] << std::endl;
        game->advanceAction();
    }
    game->close();
    delete game;
}
