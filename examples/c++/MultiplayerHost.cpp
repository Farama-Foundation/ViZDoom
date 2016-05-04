#include "ViZDoom.h"
#include <iostream>
#include <vector>

using namespace vizdoom;

int main(){

    std::cout << "\n\nMULTIPLAYER HOST EXAMPLE\n\n";


    DoomGame* game = new DoomGame();

    game->loadConfig("../../examples/config/multi.cfg");

    game->setDoomGamePath("../../scenarios/freedoom2.wad");
    //game->setDoomGamePath("../../scenarios/doom2.wad");    // Not provided with environment due to licences.

    // Host game.
    game->addGameArgs("-host 2 -deathmatch +map map01");

    game->setMode(ASYNC_SPECTATOR);         // Multiplayer requires the use of asynchronous modes.
    game->init();


    while(!game->isEpisodeFinished()){      // Play until the game (episode) is over.

        if(game->isPlayerDead()){           // Check if player is dead
            game->respawnPlayer();          // Use this to respawn immediately after death, new state will be available.

            // Or observe the game until automatic respawn.
            //game->advanceAction();
            //continue;
        }

        game->advanceAction();

    }

    game->close();
    delete game;
}
