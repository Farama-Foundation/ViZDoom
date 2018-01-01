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

    std::cout << "\n\nDELTA BUTTONS EXAMPLE\n\n";


    DoomGame *game = new DoomGame();

    game->setViZDoomPath("../../bin/vizdoom");

    game->setDoomGamePath("../../bin/freedoom2.wad");
    //game->setDoomGamePath("../../bin/doom2.wad");      // Not provided with environment due to licences.

    game->setDoomMap("map01");

    game->setScreenResolution(RES_640X480);

    // Adds delta buttons that will be allowed and set the maximum allowed value (optional).
    game->addAvailableButton(MOVE_FORWARD_BACKWARD_DELTA, 10);
    game->addAvailableButton(MOVE_LEFT_RIGHT_DELTA, 5);
    game->addAvailableButton(TURN_LEFT_RIGHT_DELTA, 5);
    game->addAvailableButton(LOOK_UP_DOWN_DELTA);

    // For normal buttons (binary) all values other than 0 are interpreted as pushed.
    // For delta buttons values determine a precision/speed.
    //
    // For TURN_LEFT_RIGHT_DELTA and LOOK_UP_DOWN_DELTA value is the angle (in degrees)
    // of which the viewing angle will change.
    //
    // For MOVE_FORWARD_BACKWARD_DELTA, MOVE_LEFT_RIGHT_DELTA, MOVE_UP_DOWN_DELTA (rarely used)
    // value is the speed of movement in a given direction (100 is close to the maximum speed).
    std::vector<double> action = {100, 10, 10, 1};

    // If button's absolute value > max button's value then value = max value with original value sign.

    // Delta buttons in spectator modes correspond to mouse movements.
    // Maximum allowed values also apply to spectator modes.
    //game->addGameArgs("+freelook 1");     //Use this to enable looking around with the mouse.
    //game->setMode(SPECTATOR);

    game->setWindowVisible(true);
    game->init();

    std::srand(time(0));

    int episodes = 10;
    unsigned int sleepTime = 28;

    // Use this to remember last shaping reward value.
    double lastTotalShapingReward = 0;

    for (int i = 0; i < episodes; ++i) {

        std::cout << "Episode #" << i + 1 << "\n";
        game->newEpisode();

        while (!game->isEpisodeFinished()) {

            GameStatePtr state = game->getState();
            game->makeAction(action);

            unsigned int time = game->getEpisodeTime();

            action[0] = time % 100 - 50;
            action[1] = time % 100 - 50;
            action[2] = time % 100 - 50;

            if(!time % 50) action[3] = -action[3];

            std::cout << "State #" << state->number << "\n";
            std::cout << "Action made:";
            for(auto a: action) std::cout << " " << a;
            std::cout << "\n";
            std::cout << "=====================\n";

            if(sleepTime) sleep(sleepTime);

        }
    }

    // It will be done automatically in destructor but after close You can init it again with different settings.
    game->close();
    delete game;
}

