#include "ViziaDoomGame.h"
#include <iostream>
#include <vector>

using namespace Vizia;

int main(){

    DoomGame* game= new DoomGame();

    std::cout << "\n\nSIMPLE EXAMPLE\n\n";

    game->setDoomEnginePath("./viziazdoom");
    game->setDoomGamePath("../scenarios/doom.wad");
    //game->setDoomGamePath("../scenarios/freedoom.wad");
    
    game->setDoomScenarioPath("../scenarios/simple.wad");
    game->setDoomMap("map01");
    game->setEpisodeTimeout(200);
    game->setLivingReward(-1);

    game->setScreenResolution(RES_320X240);

    game->setRenderHud(false);
    game->setRenderCrosshair(false);
    game->setRenderWeapon(true);
    game->setRenderDecals(false);
    game->setRenderParticles(false);

    game->setWindowVisible(true);

    game->addAvailableButton(MOVE_FORWARD);
    game->addAvailableButton(MOVE_BACKWARD);
    game->addAvailableButton(TURN_LEFT);
    game->addAvailableButton(TURN_RIGHT);
    game->addAvailableButton(ATTACK);

    game->addAvailableGameVariable(HEALTH);
    game->addAvailableGameVariable(ARMOR);
    game->addAvailableGameVariable(AMMO2);
    game->addAvailableGameVariable(AMMO3);


    game->init();


    std::vector<int> action(3);

    action[0] = 0;
    action[1] = 0;
    action[2] = 1;

    int iterations = 10000;
    int ep=1;
    for(int i = 0;i<iterations; ++i){

        if( game->isEpisodeFinished() ){
            game->newEpisode();
        }

        DoomGame::State s = game->getState();

        std::cout << "STATE NUMBER: " << s.number << " HP: " << s.gameVariables[0] << " AMMO: " << s.gameVariables[1] << std::endl;

        game->makeAction(action, 4);

        std::cout<<"reward: "<<game->getLastReward()<<std::endl;
    }
    game->close();
    delete game;
}
