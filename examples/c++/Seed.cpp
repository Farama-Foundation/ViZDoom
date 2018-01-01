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

    std::cout << "\n\nSEED EXAMPLE\n\n";


    DoomGame *game = new DoomGame();

    // Choose scenario config file you wish to be watched by agent.
    // Don't load two configs cause the second will overwrite the first one.
    // Multiple config files are ok but combining these ones doesn't make much sense.

    game->loadConfig("../../scenarios/basic.cfg");
    // game->loadConfig("../../scenarios/deadly_corridor.cfg");
    // game->loadConfig("../../scenarios/deathmatch.cfg");
    // game->loadConfig("../../scenarios/defend_the_center.cfg");
    // game->loadConfig("../../scenarios/defend_the_line.cfg");
    // game->loadConfig("../../scenarios/health_gathering.cfg");
    // game->loadConfig("../../scenarios/my_way_home.cfg");
    // game->loadConfig("../../scenarios/predict_position.cfg");
    // game->loadConfig("../../scenarios/take_cover.cfg");

    game->setDoomGamePath("../../bin/freedoom2.wad");
    //game->setDoomGamePath("../../bin/doom2.wad");      // Not provided with environment due to licences.

    game->setScreenResolution(RES_640X480);

    // Lets make episode shorter and observe starting position of Cacodemon.
    game->setEpisodeTimeout(50);

    unsigned int seed = 666;
    // Sets the seed. It could be after init as well.
    game->setSeed(seed);

    game->init();

    // Define some actions.
    std::vector<double> actions[3];
    actions[0] = {1, 0, 0};
    actions[1] = {0, 1, 0};
    actions[2] = {0, 0, 1};

    std::srand(time(0));

    int episodes = 10;
    unsigned int sleepTime = 28;

    for (int i = 0; i < episodes; ++i) {

        std::cout << "Episode #" << i + 1 << "\n";

        // Seed can be changed anytime. It will affect next episodes.
        // game->setSeed(seed);
        game->newEpisode();

        while (!game->isEpisodeFinished()) {

            // Get the state
            GameStatePtr state = game->getState();

            // Make random action and get reward
            double reward = game->makeAction(actions[std::rand() % 3]);

            std::cout << "State #" << state->number << "\n";
            std::cout << "Action reward: " << reward << "\n";
            std::cout << "Seed: " << game->getSeed() << "\n";
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

