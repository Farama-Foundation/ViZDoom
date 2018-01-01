#include "ViZDoom.h"
#include <iostream>
#include <vector>

using namespace vizdoom;

int main(){

    std::cout << "\n\nCIG EXAMPLE\n\n";


    DoomGame* game = new DoomGame();

    // Use CIG example config or Your own.
    game->loadConfig("../../scenarios/cig.cfg");

    // Select game and map You want to use.
    game->setDoomGamePath("../../bin/freedoom2.wad");
    //game->setDoomGamePath("../../bin/doom2.wad");      // Not provided with environment due to licences.

    game->setDoomMap("map01");      // Limited deathmatch.
    //game->setDoomMap("map02");      // Full deathmatch.

    // Join existing game.
    game->addGameArgs("-join 127.0.0.1");   // Connect to a host for a multiplayer game.

    // Name your agent and select color
    // colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
    game->addGameArgs("+name AI +colorset 0");

    game->setMode(ASYNC_PLAYER);
    game->init();

    while(!game->isEpisodeFinished()){      // Play until the game (episode) is over.

        GameStatePtr state = game->getState();
        // Analyze the state.

        std::vector<double> action(game->getAvailableButtonsSize());
        // Set your action.

        game->makeAction(action);

        if(game->isPlayerDead()){           // Check if player is dead
            game->respawnPlayer();          // Use this to respawn immediately after death, new state will be available.
        }

        std::cout << game->getEpisodeTime() << " Frags: " << game->getGameVariable(FRAGCOUNT) << std::endl;
    }

    game->close();
}
