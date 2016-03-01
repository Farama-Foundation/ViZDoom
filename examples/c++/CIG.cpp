#include "ViziaDoomGame.h"
#include <iostream>
#include <vector>

using namespace Vizia;

int main(){

    DoomGame* game = new DoomGame();

    std::cout << "CIG EXAMPLE\n\n";

    //Use CIG example config or Your own.
    game->loadConfig("../../examples/config/cig.cfg");
    game->setDoomMap("map01");
    //game->setDoomMap("map02");

    //Join existing game.
    game->addGameArgs("-join 127.0.0.1");     //Connect to a host for a multiplayer game.

    //Name Your AI.
    game->addGameArgs("+name AI");

    game->setMode(ASYNC_PLAYER);                //Multiplayer requires the use of asynchronous modes.
    game->init();

    while(!game->isEpisodeFinished()){          //Play until the game (episode) is over.

        if(game->isPlayerDead()){
            game->respawnPlayer();              //Use this to respawn immediately after death.

            //Or observe the game until automatic respawn.
            //game->advanceAction();
            //continue;
        }

        DoomGame::State state = game->getState();
        //Analyze the state.

        std::vector<int> action(game->getAvailableButtonsSize());
        //Set your action.

        game->makeAction(action);

        std::cout << game->getEpisodeTime() << " Frags: " << game.getGameVariable(FRAGCOUNT) << std::endl;
    }

    game->close();
}
