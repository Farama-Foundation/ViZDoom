#include "ViZDoom.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace vizdoom;

int main(){

    std::cout << "\n\nMULTIPLAYER EXAMPLE\n\n";


    DoomGame* game = new DoomGame();

    game->loadConfig("../../examples/config/multi.cfg");

    game->setDoomGamePath("../../scenarios/freedoom2.wad");
    //game->setDoomGamePath("../../scenarios/doom2.wad");      // Not provided with environment due to licences.

    // Join existing game (see MultiplayerHost.cpp example)
    game->addGameArgs("-join 127.0.0.1");   // Connect to a host for a multiplayer game.

    game->setMode(ASYNC_PLAYER);            // Multiplayer requires the use of asynchronous modes.
    game->init();


    std::vector<int> actions[3];
    int action0[] = {1, 0, 0};
    actions[0] = std::vector<int>(action0, action0 + sizeof(action0) / sizeof(int));

    int action1[] = {0, 1, 0};
    actions[1] = std::vector<int>(action1, action1 + sizeof(action1) / sizeof(int));

    int action2[] = {0, 0, 1};
    actions[2] = std::vector<int>(action2, action2 + sizeof(action2) / sizeof(int));

    std::srand(time(0));

    while(!game->isEpisodeFinished()){      // Play until the game (episode) is over.

        if(game->isPlayerDead()){           // Check if player is dead
            game->respawnPlayer();          // Use this to respawn immediately after death, new state will be available.

            // Or observe the game until automatic respawn.
            //game->advanceAction();
            //continue;
        }

        // Get the state
        GameState s = game->getState();

        // Make random action and get reward
        double r = game->makeAction(actions[std::rand() % 3]);

        std::cout << "State #" << s.number << "\n";
        std::cout << "Action reward: " << r <<"\n";
        std::cout << "Frags: " << game->getGameVariable(FRAGCOUNT) << std::endl;
        std::cout << "=====================\n";

    }

    game->close();
    delete game;
}
