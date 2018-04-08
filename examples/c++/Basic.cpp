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

int main() {

    std::cout << "\n\nBASIC EXAMPLE\n\n";


    // Create DoomGame instance. It will run the game and communicate with you.
    DoomGame *game = new DoomGame();

    // Sets path to vizdoom engine executive which will be spawned as a separate process. Default is "./vizdoom".
    game->setViZDoomPath("../../bin/vizdoom");

    // Sets path to doom2 iwad resource file which contains the actual doom game-> Default is "./doom2.wad".
    game->setDoomGamePath("../../bin/freedoom2.wad");
    //game->setDoomGamePath("../../bin/doom2.wad");      // Not provided with environment due to licences.

    // Sets path to additional resources iwad file which is basically your scenario iwad.
    // If not specified default doom2 maps will be used and it's pretty much useless... unless you want to play doom.
    game->setDoomScenarioPath("../../scenarios/basic.wad");

    // Set map to start (scenario .wad files can contain many maps).
    game->setDoomMap("map01");

    // Sets resolution. Default is 320X240
    game->setScreenResolution(RES_640X480);

    // Sets the screen buffer format. Not used here but now you can change it. Default is CRCGCB.
    game->setScreenFormat(RGB24);

    // Sets other rendering options
    game->setRenderHud(false);
    game->setRenderMinimalHud(false); // If hud is enabled
    game->setRenderCrosshair(false);
    game->setRenderWeapon(true);
    game->setRenderDecals(false);
    game->setRenderParticles(false);
    game->setRenderEffectsSprites(false);
    game->setRenderMessages(false);
    game->setRenderCorpses(false);

    // Adds buttons that will be allowed.
    game->addAvailableButton(MOVE_LEFT);
    game->addAvailableButton(MOVE_RIGHT);
    game->addAvailableButton(ATTACK);

    // Adds game variables that will be included in state.
    game->addAvailableGameVariable(AMMO2);

    // Causes episodes to finish after 200 tics (actions)
    game->setEpisodeTimeout(200);

    // Makes episodes start after 10 tics (~after raising the weapon)
    game->setEpisodeStartTime(10);

    // Makes the window appear (turned on by default)
    game->setWindowVisible(true);

    // Turns on the sound. (turned off by default)
    game->setSoundEnabled(true);

    // Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
    game->setMode(PLAYER);

    // Enables engine output to console.
    //game->setConsoleEnabled(true);

    // Initialize the game. Further configuration won't take any effect from now on.
    game->init();


    // Define some actions. Each list entry corresponds to declared buttons:
    // MOVE_LEFT, MOVE_RIGHT, ATTACK
    // game.getAvailableButtonsSize() can be used to check the number of available buttons.
    // more combinations are naturally possible but only 3 are included for transparency when watching.
    std::vector<double> actions[3];
    actions[0] = {1, 0, 0};
    actions[1] = {0, 1, 0};
    actions[2] = {0, 0, 1};

    std::srand(time(0));

    // Run this many episodes
    int episodes = 10;

    // Sets time that will pause the engine after each action.
    // Without this everything would go too fast for you to keep track of what's happening.
    unsigned int sleepTime = 1000 / DEFAULT_TICRATE; // = 28

    for (int i = 0; i < episodes; ++i) {

        std::cout << "Episode #" << i + 1 << "\n";

        // Starts a new episode. It is not needed right after init() but it doesn't cost much and the loop is nicer.
        game->newEpisode();

        while (!game->isEpisodeFinished()) {

            // Get the state
            GameStatePtr state = game->getState(); // GameStatePtr is std::shared_ptr<GameState>

            // Which consists of:
            unsigned int n              = state->number;
            std::vector<double> vars       = state->gameVariables;
            BufferPtr screenBuf         = state->screenBuffer;
            BufferPtr depthBuf          = state->depthBuffer;
            BufferPtr labelsBuf         = state->labelsBuffer;
            BufferPtr automapBuf        = state->automapBuffer;
            // BufferPtr is std::shared_ptr<Buffer> where Buffer is std::vector<uint8_t>
            std::vector<Label> labels   = state->labels;

            // Make random action and get reward
            double reward = game->makeAction(actions[std::rand() % game->getAvailableButtonsSize()]);

            // You can also get last reward by using this function
            // double reward = game->getLastReward();

            // Makes a "prolonged" action and skip frames.
            //int skiprate = 4
            //double reward = game.makeAction(choice(actions), skiprate)

            // The same could be achieved with:
            //game.setAction(choice(actions))
            //game.advanceAction(skiprate)
            //reward = game.getLastReward()

            std::cout << "State #" << n << "\n";
            std::cout << "Game variables: " << vars[0] << "\n";
            std::cout << "Action reward: " << reward << "\n";
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
        
