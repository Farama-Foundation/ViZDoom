#include "ViZDoom.h"
#include <iostream>
#include <vector>

using namespace vizdoom;

int main(){

    std::cout << "\n\nCIG HOST EXAMPLE\n\n";


    DoomGame* game = new DoomGame();

    // Use CIG example config or Your own.
    game->loadConfig("../../examples/config/cig.cfg");

    // Select game and map You want to use.
    game->setDoomGamePath("../../scenarios/freedoom2.wad");
    //game->setDoomGamePath("../../scenarios/doom2.wad");      // Not provided with environment due to licences.

    game->setDoomMap("map01");      // Limited deathmatch.
    //game->setDoomMap("map02");      // Full deathmatch.

    // Host game with options that will be used in the competition.
    game->addGameArgs("-host 8 "                // This machine will function as a host for a multiplayer game with this many players (including this machine). It will wait for other machines to connect using the -join parameter and then start the game when everyone is connected.
                      "-deathmatch "            // Deathmatch rules are used for the game.
                      "+timelimit 10.0 "        // The game (episode) will end after this many minutes have elapsed.
                      "+sv_forcerespawn 1 "     // Players will respawn automatically after they die.
                      "+sv_noautoaim 1 "        // Autoaim is disabled for all players.
                      "+sv_respawnprotect 1 "   // Players will be invulnerable for two second after spawning.
                      "+sv_spawnfarthest 1 "    // Players will be spawned as far as possible from any other players.
                      "+vizdoom_nocheat 1");    // Disables depth buffer and the ability to use commands that could interfere with multiplayer game.

    // Name Your AI.
    game->addGameArgs("+name AI");

    game->setMode(ASYNC_PLAYER);                // Multiplayer requires the use of asynchronous modes.
    game->init();

    while(!game->isEpisodeFinished()){          // Play until the game (episode) is over.

        if(game->isPlayerDead()){
            game->respawnPlayer();              // Use this to respawn immediately after death, new state will be available.

            // Or observe the game until automatic respawn.
            //game->advanceAction();
            //continue;
        }

        GameState state = game->getState();
        // Analyze the state.

        std::vector<int> action(game->getAvailableButtonsSize());
        // Set your action.

        game->makeAction(action);

        std::cout << game->getEpisodeTime() << " Frags: " << game->getGameVariable(FRAGCOUNT) << std::endl;
    }

    game->close();
}
