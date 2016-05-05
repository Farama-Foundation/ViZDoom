#include "ViZDoom.h"
#include <iostream>
#include <vector>

using namespace vizdoom;

int main(){

    std::cout << "\n\nCIG EXAMPLE\n\n";


    DoomGame* game = new DoomGame();

    // Use CIG example config or Your own.
    game->loadConfig("../../examples/config/cig.cfg");

    // Select game and map You want to use.
    game->setDoomGamePath("../../scenarios/freedoom2.wad");
    //game->setDoomGamePath("../../scenarios/doom2.wad");      // Not provided with environment due to licences.

    game->setDoomMap("map01");      // Limited deathmatch.
    //game->setDoomMap("map02");      // Full deathmatch.

    // Join existing game.
    game->addGameArgs("-join 127.0.0.1");       // Connect to a host for a multiplayer game.

    // Name Your AI.
    game->addGameArgs("+name AI");

    game->setMode(ASYNC_PLAYER);                // Multiplayer requires the use of asynchronous modes.
    game->init();

    while(!game->isEpisodeFinished()){          // Play until the game (episode) is over.

        if(game->isPlayerDead()){               // Check if player is dead
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
