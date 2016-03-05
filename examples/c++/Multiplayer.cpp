#include "ViZDoomGame.h"
#include <iostream>
#include <vector>

using namespace vizdoom;

int main(){

    DoomGame* game = new DoomGame();

    std::cout << "MULTIPLAYER EXAMPLE\n\n";

    // Use CIG example config or Your own.
    game->loadConfig("../../examples/config/multi.cfg");

    game->setDoomGamePath("../../scenarios/freedoom2.wad");
    // game->setDoomGamePath("../../scenarios/doom2.wad");     // Not provided with environment due to licences.

    // Host game.
    game.addGameArgs("-host 2 -deathmatch +map map01");

    // Or join existing game.
    // game->addGameArgs("-join 127.0.0.1");       // Connect to a host for a multiplayer game.

    game->setMode(ASYNC_PLAYER);                // Multiplayer requires the use of asynchronous modes.
    game->init();


    std::vactor<int> actions[3];
    int action[] = {1, 0, 0};
    actions[0] = std::vector<int>(action, action + sizeof(action) / sizeof(int));

    int action[] = {0, 1, 0};
    actions[1] = std::vector<int>(action, action + sizeof(action) / sizeof(int));

    int action[] = {0, 0, 1};
    actions[2] = std::vector<int>(action, action + sizeof(action) / sizeof(int));

    std::srand(time());

    while(!game->isEpisodeFinished()){          // Play until the game (episode) is over.

        if(game->isPlayerDead()){               // Check if player is dead
            game->respawnPlayer();              // Use this to respawn immediately after death, new state will be available.

            // Or observe the game until automatic respawn.
            // game->advanceAction();
            // continue;
        }

        // Get the state
        GameState state = game->getState();

        // Make random action and get reward
        double r = game->makeAction(actions[std::rand() % 3]);

        std::cout << "State #" << s.number << "\n";
        std::cout << "Action reward: " << r <<"\n";
        std::cout << "Frags: " << game.getGameVariable(FRAGCOUNT) << std::endl;
        std::cout << "=====================\n");

    }

    game->close();
    delete game;
}
