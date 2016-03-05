#include "ViZDoomGame.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#inlucde <ctime>

using namespace ViZDoom;

int main(){

    std::cout << "\n\nDELTA BUTTONS EXAMPLE\n\n";

    DoomGame *game = new DoomGame();

    game->setDoomEnginePath("../../bin/vizdoom");

    game->setDoomGamePath("../../scenarios/freedoom2.wad");
    // game->setDoomGamePath("../../scenarios/doom2.wad");   # Not provided with environment due to licences.

    game->setDoomMap("map01");

    game->setScreenResolution(ScreenResolution.RES_640X480);

    // Adds delta buttons that will be allowed and set the maximum allowed value (optional).
    game->addAvailableButton(Button.MOVE_FORWARD_BACKWARD_DELTA, 50);
    game->addAvailableButton(Button.MOVE_LEFT_RIGHT_DELTA, 20);
    game->addAvailableButton(Button.TURN_LEFT_RIGHT_DELTA);
    game->addAvailableButton(Button.LOOK_UP_DOWN_DELTA);

    // For normal buttons (binary) all values other than 0 are interpreted as pushed.
    // For delta buttons values determine a precision/speed.
    //
    // For TURN_LEFT_RIGHT_DELTA and LOOK_UP_DOWN_DELTA value is the angle (in degrees)
    // of which the viewing angle will change.
    //
    // For MOVE_FORWARD_BACKWARD_DELTA, MOVE_LEFT_RIGHT_DELTA, MOVE_UP_DOWN_DELTA (rarely used)
    // value is the speed of movement in a given direction (100 is close to the maximum speed).
    std::vactor<int> actions[2];
    int action[] = {100, 10, 10, 10};
    actions[0] = std::vector<int>(action, action + sizeof(action) / sizeof(int));

    int action[] = {20, -30, -20, -15};
    actions[1] = std::vector<int>(action, action + sizeof(action) / sizeof(int));

    // If button's absolute value > max button's value then value = max value with original value sign.

    // Delta buttons in spectator modes correspond to mouse movements.
    // Maximum allowed values also apply to spectator modes.
    // game->addGameArgs("+freelook 1");    //Use this to enable look around with the mouse.
    // game->setMode(SPECTATOR);

    game->setWindowVisible(True);
    game->init();

    std::srand(time());

    // Run this many episodes.
    int episodes = 10;

    // Use this to remember last shaping reward value.
    double lastSummaryShapingReward = 0;

    for (int i = 0; i < episodes; ++i) {

        std::cout << "Episode #" << i + 1 << "\n";
        game->newEpisode();

        while (!game->isEpisodeFinihsed()) {

            // Get the state
            GameState s = game->getState();

            // Make random action and get reward
            game->makeAction(actions[std::rand() % 3]);

        }
    }

    // It will be done automatically in destructor but after close You can init it again with different settings.
    game->close();
    delete game;
}

