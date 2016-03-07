#include "ViZDoomGame.h"
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

    game->loadConfig("../config/basic.cfg");
    // game->loadConfig("../config/deadly_corridor.cfg");
    // game->loadConfig("../config/deathmatch.cfg");
    // game->loadConfig("../config/defend_the_center.cfg");
    // game->loadConfig("../config/defend_the_line.cfg");
    // game->loadConfig("../config/health_gathering.cfg");
    // game->loadConfig("../config/my_way_home.cfg");
    // game->loadConfig("../config/predict_position.cfg");
    // game->loadConfig("../config/take_cover.cfg");

    game->set_screen_resolution(ScreenResolution.RES_640X480);

    unsigned int seed = 1234;
    // Sets the seed. It could be after init as well.
    game->setSeed(seed);

    game->init();

    std::vactor<int> actions[3];
    int action[] = {1, 0, 0};
    actions[0] = std::vector<int>(action, action + sizeof(action) / sizeof(int));

    int action[] = {0, 1, 0};
    actions[1] = std::vector<int>(action, action + sizeof(action) / sizeof(int));

    int action[] = {0, 0, 1};
    actions[2] = std::vector<int>(action, action + sizeof(action) / sizeof(int));

    std::srand(time());

    // Run this many episodes
    int episodes = 10;

    for (int i = 0; i < episodes; ++i) {

        std::cout << "Episode #" << i + 1 << "\n";

        // Seed can be changed anytime. It will affect next episodes.
        // game->setSeed(seed);
        game->newEpisode();

        while (!game->isEpisodeFinihsed()) {

            // Get the state
            GameState s = game->getState();

            // Make random action and get reward
            game->makeAction(actions[std::rand() % 3]);

            std::cout << "State #" << s.number << "\n";
            std::cout << "Action reward: " << r << "\n";
            std::cout << "Seed: " << game->getSeed() << "\n";
            std::cout << "=====================\n";

        }

        std::cout << "Episode finished.\n";
        std::cout << "Summary reward: " << game->getSummaryReward() << "\n";
        std::cout << "************************\n";

    }

    // It will be done automatically in destructor but after close You can init it again with different settings.
    game->close();
    delete game;
}

