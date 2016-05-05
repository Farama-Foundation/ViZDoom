#include "ViZDoom.h"
#include <iostream>
#include <vector>

using namespace vizdoom;

int main(){

    std::cout << "\n\nCIG BOTS EXAMPLE\n\n";


    DoomGame* game = new DoomGame();

    // Use CIG example config or Your own.
    game->loadConfig("../../examples/config/cig.cfg");

    // Select game and map You want to use.
    game->setDoomGamePath("../../scenarios/freedoom2.wad");
    //game->setDoomGamePath("../../scenarios/doom2.wad");      // Not provided with environment due to licences.

    game->setDoomMap("map01");      // Limited deathmatch.
    //game->setDoomMap("map02");      // Full deathmatch.

    // Start multiplayer game only with Your AI (with options that will be used in the competition, details in CIGHost example).
    game->addGameArgs("-host 1 -deathmatch +timelimit 10.0 "
                      "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1");

    // Name Your AI.
    game->addGameArgs("+name AI");

    // Multiplayer requires the use of asynchronous modes, but when playing only with bots, synchronous modes can also be used.
    game->setMode(ASYNC_PLAYER);
    game->init();

    // Add bots (file examples/bots.cfg must be placed in the same directory as the Doom executable file).
    for(int i=0; i < 7; ++i) {
        game->sendGameCommand("addbot");
    }

    while(!game->isEpisodeFinished()){          // Play until the game (episode) is over.

        if(game->isPlayerDead()){
            game->respawnPlayer();              // Use this to respawn immediately after death, new state will be available.

            // Or observe the game until automatic respawn.
            //game->advanceAction();
            //continue;
        }

        GameState state = game->getState();
        // Analyze the state.

        std::vector<int> action(game->getAvailableButtonsSize());
        // Set your action.

        game->makeAction(action);

        std::cout << game->getEpisodeTime() << " Frags: " << game->getGameVariable(FRAGCOUNT) << std::endl;
    }

    game->close();
}
