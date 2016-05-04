#include "ViZDoom.h"
#include <iostream>
#include <vector>

using namespace vizdoom;

int main(){

    std::cout << "\n\nSPECTATOR EXAMPLE\n\n";


    DoomGame *game = new DoomGame();

    // Choose scenario config file you wish to be watched by agent.
    // Don't load two configs cause the second will overwrite the first one.
    // Multiple config files are ok but combining these ones doesn't make much sense.

    //game->loadConfig("../../examples/config/basic.cfg");
    //game->loadConfig("../../examples/config/deadly_corridor.cfg");
    game->loadConfig("../../examples/config/deathmatch.cfg");
    //game->loadConfig("../../examples/config/defend_the_center.cfg");
    //game->loadConfig("../../examples/config/defend_the_line.cfg");
    //game->loadConfig("../../examples/config/health_gathering.cfg");
    //game->loadConfig("../../examples/config/my_way_home.cfg");
    //game->loadConfig("../../examples/config/predict_position.cfg");
    //game->loadConfig("../../examples/config/take_cover.cfg");

    game->setScreenResolution(RES_640X480);

    // Enables spectator mode, so You can play and agent watch your actions.
    // You can only use the buttons selected as available.
    game->setMode(SPECTATOR);

    game->init();

    // Run this many episodes
    int episodes = 10;

    for (int i = 0; i < episodes; ++i) {

        std::cout << "Episode #" << i + 1 << "\n";

        // Starts a new episode. It is not needed right after init() but it doesn't cost much and the loop is nicer.
        game->newEpisode();

        while (!game->isEpisodeFinished()) {

            // Get the state.
            GameState s = game->getState();

            // Advances action - lets You play next game tic.
            game->advanceAction();

            // You can also advance a few tics at once.
            // game->advanceAction(4);

            // Get the last action performed by You.
            std::vector<int> a = game->getLastAction();

            // And reward You get.
            double r = game->getLastReward();

            std::cout << "State #" << s.number << "\n";
            std::cout << "Action made: ";
            for(int i = 0; i < a.size(); ++i) std::cout << " " << a[i];
            std::cout <<"\n";
            std::cout << "Action reward: " << r <<"\n";
            std::cout << "=====================\n";

        }

        std::cout << "Episode finished.\n";
        std::cout << "Total reward: " << game->getTotalReward() << "\n";
        std::cout << "************************\n";
    }

    // It will be done automatically in destructor but after close You can init it again with different settings.
    game->close();
    delete game;
}
