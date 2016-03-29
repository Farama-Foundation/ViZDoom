#include "ViZDoom.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace vizdoom;

int main(){

    std::cout << "\n\nSEED EXAMPLE\n\n";


    DoomGame *game = new DoomGame();

    // Choose scenario config file you wish to be watched by agent.
    // Don't load two configs cause the second will overwrite the first one.
    // Multiple config files are ok but combining these ones doesn't make much sense.

    game->loadConfig("../../examples/config/basic.cfg");
    // game->loadConfig("../../examples/config/deadly_corridor.cfg");
    // game->loadConfig("../../examples/config/deathmatch.cfg");
    // game->loadConfig("../../examples/config/defend_the_center.cfg");
    // game->loadConfig("../../examples/config/defend_the_line.cfg");
    // game->loadConfig("../../examples/config/health_gathering.cfg");
    // game->loadConfig("../../examples/config/my_way_home.cfg");
    // game->loadConfig("../../examples/config/predict_position.cfg");
    // game->loadConfig("../../examples/config/take_cover.cfg");

    game->setScreenResolution(RES_640X480);

    unsigned int seed = 1234;
    // Sets the seed. It could be after init as well.
    game->setSeed(seed);

    game->init();


    std::vector<int> actions[3];
    int action0[] = {1, 0, 0};
    actions[0] = std::vector<int>(action0, action0 + sizeof(action0) / sizeof(int));

    int action1[] = {0, 1, 0};
    actions[1] = std::vector<int>(action1, action1 + sizeof(action1) / sizeof(int));

    int action2[] = {0, 0, 1};
    actions[2] = std::vector<int>(action2, action2 + sizeof(action2) / sizeof(int));

    std::srand(time(0));

    // Run this many episodes
    int episodes = 10;

    for (int i = 0; i < episodes; ++i) {

        std::cout << "Episode #" << i + 1 << "\n";

        // Seed can be changed anytime. It will affect next episodes.
        // game->setSeed(seed);
        game->newEpisode();

        while (!game->isEpisodeFinished()) {

            // Get the state
            GameState s = game->getState();

            // Make random action and get reward
            double r = game->makeAction(actions[std::rand() % 3]);

            std::cout << "State #" << s.number << "\n";
            std::cout << "Action reward: " << r << "\n";
            std::cout << "Seed: " << game->getSeed() << "\n";
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

