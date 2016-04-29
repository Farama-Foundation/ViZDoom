#include "ViZDoom.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace vizdoom;

int main(){

    std::cout << "\n\nDELTA BUTTONS EXAMPLE\n\n";


    DoomGame *game = new DoomGame();

    game->setViZDoomPath("../../bin/vizdoom");

    game->setDoomGamePath("../../scenarios/freedoom2.wad");
    //game->setDoomGamePath("../../scenarios/doom2.wad");      // Not provided with environment due to licences.

    game->setDoomMap("map01");

    game->setScreenResolution(RES_640X480);

    // Adds delta buttons that will be allowed and set the maximum allowed value (optional).
    game->addAvailableButton(MOVE_FORWARD_BACKWARD_DELTA, 5);
    game->addAvailableButton(MOVE_LEFT_RIGHT_DELTA, 2);
    game->addAvailableButton(TURN_LEFT_RIGHT_DELTA);
    game->addAvailableButton(LOOK_UP_DOWN_DELTA);

    // For normal buttons (binary) all values other than 0 are interpreted as pushed.
    // For delta buttons values determine a precision/speed.
    //
    // For TURN_LEFT_RIGHT_DELTA and LOOK_UP_DOWN_DELTA value is the angle (in degrees)
    // of which the viewing angle will change.
    //
    // For MOVE_FORWARD_BACKWARD_DELTA, MOVE_LEFT_RIGHT_DELTA, MOVE_UP_DOWN_DELTA (rarely used)
    // value is the speed of movement in a given direction (100 is close to the maximum speed).
    std::vector<int> actions[2];
    int action0[] = {10, 1, 1, 1};
    actions[0] = std::vector<int>(action0, action0 + sizeof(action0) / sizeof(int));

    int action1[] = {2, -3, -2, 0};
    actions[1] = std::vector<int>(action1, action1 + sizeof(action1) / sizeof(int));

    // If button's absolute value > max button's value then value = max value with original value sign.

    // Delta buttons in spectator modes correspond to mouse movements.
    // Maximum allowed values also apply to spectator modes.
    //game->addGameArgs("+freelook 1");     //Use this to enable looking around with the mouse.
    //game->setMode(SPECTATOR);

    game->setWindowVisible(true);
    game->init();

    std::srand(time(0));

    // Run this many episodes.
    int episodes = 10;

    // Use this to remember last shaping reward value.
    double lastTotalShapingReward = 0;

    for (int i = 0; i < episodes; ++i) {

        std::cout << "Episode #" << i + 1 << "\n";
        game->newEpisode();

        game->getEpisodeTime();

        while (!game->isEpisodeFinished()) {

            // Get the state
            GameState s = game->getState();

            // Make random action and get reward
            game->makeAction(actions[std::rand() % 2]);

        }
    }

    // It will be done automatically in destructor but after close You can init it again with different settings.
    game->close();
    delete game;
}

