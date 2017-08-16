import vizdoom.*;

import java.util.*;
import java.lang.*;

public class CIGHost {

    public static void main (String[] args) {

        System.out.println("\n\nCIG HOST EXAMPLE\n");


        DoomGame game = new DoomGame();

        // Use CIG example config or Your own.
        game.loadConfig("../../scenarios/cig.cfg");

        // Select game and map You want to use.
        game.setDoomGamePath("../../bin/freedoom2.wad");
        //game.setDoomGamePath("../../bin/doom2.wad");   // Not provided with environment due to licences.

        game.setDoomMap("map01");      // Limited deathmatch.
        //game.setDoomMap("map02");      // Full deathmatch.

        // Host game with options that will be used in the competition.
        game.addGameArgs("-host 8 "                // This machine will function as a host for a multiplayer game with this many players (including this machine). It will wait for other machines to connect using the -join parameter and then start the game when everyone is connected.
                        + "-deathmatch "           // Deathmatch rules are used for the game.
                        + "+timelimit 10.0 "       // The game (episode) will end after this many minutes have elapsed.
                        + "+sv_forcerespawn 1 "    // Players will respawn automatically after they die.
                        + "+sv_noautoaim 1 "       // Autoaim is disabled for all players.
                        + "+sv_respawnprotect 1 "  // Players will be invulnerable for two second after spawning.
                        + "+sv_spawnfarthest 1 "   // Players will be spawned as far as possible from any other players.
                        + "+sv_nocrouch 1 "        // Disables crouching.
                        + "+viz_respawn_delay 10 " // Sets delay between respanws (in seconds).
                        + "+viz_nocheat 1");       // Disables depth and labels buffer and the ability to use commands that could interfere with multiplayer game.

        // Name your agent and select color
        // colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
        game.addGameArgs("+name AI +colorset 0");

        game.setMode(Mode.ASYNC_PLAYER);
        game.init();

        while(!game.isEpisodeFinished()){       // Play until the game (episode) is over.

            GameState state = game.getState();
            // Analyze the state.

            double[] action= new double[game.getAvailableButtonsSize()];
            // Set your action.

            game.makeAction(action);

            if(game.isPlayerDead()){            // Check if player is dead
                game.respawnPlayer();           // Use this to respawn immediately after death, new state will be available.
            }

            System.out.println(game.getEpisodeTime() + " Frags: " + game.getGameVariable(GameVariable.FRAGCOUNT));
        }

        game.close();
    }
}
