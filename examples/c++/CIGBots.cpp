#include "ViziaDoomGame.h"
#include <iostream>
#include <vector>

using namespace Vizia;

int main(){

    DoomGame* game = new DoomGame();

    std::cout << "CIG BOTS EXAMPLE\n\n";

    //Use CIG example config or Your own.
    game->loadConfig("../../examples/config/cig.cfg");
    game->setDoomMap("map01");
    //game->setDoomMap("map02");

    //Start multiplayer game only with Your AI (with options that will be used in the competition).
    game->addGameArgs("-host 1 -deathmatch +timelimit 10.0 "
                      "+sv_forcerespawn 1 +sv_losefrag 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1");

    //Name Your AI.
    game->addGameArgs("+name AI");

    game->setMode(ASYNC_PLAYER);                //Multiplayer requires the use of asynchronous modes.
    game->init();

    //Add bots (file examples/bots.cfg must be placed in the same directory as the Doom executable file).
    for(int i=0; i < 7; ++i) {
        game->sendGameCommand("addbot");
    }

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
