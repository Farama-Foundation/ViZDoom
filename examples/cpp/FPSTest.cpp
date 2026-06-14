#include "ViZDoom.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <thread>
#include <chrono>

using namespace std::chrono;
using namespace vizdoom;


// Rendering options:
ScreenResolution resolution = RES_320X240;
ScreenFormat screenFormat = CRCGCB;
bool depthBuffer = false;
bool labelsBuffer = false;
bool automapBuffer = false;
bool audioBuffer = false;
bool objectsInfo = false;
bool sectorsInfo = false;
int iterations = 10000;

int main() {

    std::cout << "\n\nFPS TEST\n\n";

    DoomGame *game = new DoomGame();
    game->setViZDoomPath("../../bin/vizdoom");
    game->setDoomGamePath("../../bin/freedoom2.wad");

    game->loadConfig("../../scenarios/basic.cfg");

    game->setScreenResolution(resolution);
    game->setScreenFormat(screenFormat);
    game->setDepthBufferEnabled(depthBuffer);
    game->setLabelsBufferEnabled(labelsBuffer);
    game->setAutomapBufferEnabled(automapBuffer);
    game->setAudioBufferEnabled(audioBuffer);
    game->setObjectsInfoEnabled(objectsInfo);
    game->setSectorsInfoEnabled(sectorsInfo);

    game->setWindowVisible(false);

    std::vector<double> actions[3];
    actions[0] = {1, 0, 0};
    actions[1] = {0, 1, 0};
    actions[2] = {0, 0, 1};

    std::srand(time(0));

    game->init();

    std::cout << "Checking FPS with selected features. It may take some time. Be patient.\n";

    auto start = high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        if(game->isEpisodeFinished()) game->newEpisode();

        GameStatePtr state = game->getState();
        double reward = game->makeAction(actions[std::rand() % game->getAvailableButtonsSize()]);
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    game->close();
    delete game;

    std::cout << "Results:\n";
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "Duration: " << static_cast<double>(duration.count()) / 1000 << "s" << std::endl;
    std::cout << "FPS: " << static_cast<double>(iterations) / duration.count() * 1000 << std::endl;
}
