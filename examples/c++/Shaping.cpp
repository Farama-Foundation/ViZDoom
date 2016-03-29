#include "ViZDoom.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace vizdoom;

int main(){

    std::cout << "\n\nSHAPING EXAMPLE\n\n";


    DoomGame *game = new DoomGame();

    // Health gathering scenario has scripted shaping reward.
    game->loadConfig("../../examples/config/health_gathering.cfg");

    game->setScreenResolution(RES_640X480);

    game->init();


    std::vector<int> actions[3];
    int action0[] = {1, 0, 0};
    actions[0] = std::vector<int>(action0, action0 + sizeof(action0) / sizeof(int));

    int action1[] = {0, 1, 0};
    actions[1] = std::vector<int>(action1, action1 + sizeof(action1) / sizeof(int));

    int action2[] = {0, 0, 1};
    actions[2] = std::vector<int>(action2, action2 + sizeof(action2) / sizeof(int));

    std::srand(time(0));

    // Run this many episodes.
    int episodes = 10;

    // Use this to remember last shaping reward value.
    double lastTotalShapingReward = 0;

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

            // Retrieve the shaping reward
            int _ssr = game->getGameVariable(USER1);        // Get value of scripted variable
            double ssr = DoomFixedToDouble(_ssr);           // If value is in DoomFixed format project it to double
            double sr = ssr - lastTotalShapingReward;
            lastTotalShapingReward = ssr;

            std::cout << "State #" << s.number << "\n";
            std::cout << "Health: " << s.gameVariables[0] << "\n";
            std::cout << "Action reward: " << r << "\n";
            std::cout << "Action shaping reward: " << sr << "\n";
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

