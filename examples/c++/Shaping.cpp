#include "ViZDoom.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <thread>

void sleep(unsigned int time){
    std::this_thread::sleep_for(std::chrono::milliseconds(time));
}

using namespace vizdoom;

int main(){

    std::cout << "\n\nSHAPING EXAMPLE\n\n";


    DoomGame *game = new DoomGame();

    // Health gathering scenario has scripted shaping reward.
    game->loadConfig("../../scenarios/health_gathering.cfg");

    game->setDoomGamePath("../../bin/freedoom2.wad");
    //game->setDoomGamePath("../../bin/doom2.wad");      // Not provided with environment due to licences.

    game->setScreenResolution(RES_640X480);

    game->init();

    // Define some actions.
    std::vector<double> actions[3];
    actions[0] = {1, 0, 0};
    actions[1] = {0, 1, 0};
    actions[2] = {0, 0, 1};

    std::srand(time(0));

    int episodes = 10;
    unsigned int sleepTime = 28;

    // Use this to remember last shaping reward value.
    double lastTotalShapingReward = 0;

    for (int i = 0; i < episodes; ++i) {

        std::cout << "Episode #" << i + 1 << "\n";

        // Seed can be changed anytime. It will affect next episodes.
        // game->setSeed(seed);
        game->newEpisode();

        lastTotalShapingReward = 0;

        while (!game->isEpisodeFinished()) {

            // Get the state
            GameStatePtr state = game->getState();

            // Make random action and get reward
            double reward = game->makeAction(actions[std::rand() % 3]);

            // Retrieve the shaping reward
            int fixedShapingReward = game->getGameVariable(USER1);     // Get value of scripted variable
            double shapingReward = doomFixedToDouble(shapingReward);   // If value is in DoomFixed format project it to double
            shapingReward = shapingReward - lastTotalShapingReward;
            lastTotalShapingReward += shapingReward;

            std::cout << "State #" << state->number << "\n";
            std::cout << "Health: " << state->gameVariables[0] << "\n";
            std::cout << "Action reward: " << reward << "\n";
            std::cout << "Action shaping reward: " << shapingReward << "\n";
            std::cout << "=====================\n";

            if(sleepTime) sleep(sleepTime);

        }

        std::cout << "Episode finished.\n";
        std::cout << "Total reward: " << game->getTotalReward() << "\n";
        std::cout << "************************\n";

    }

    // It will be done automatically in destructor but after close You can init it again with different settings.
    game->close();
    delete game;
}

