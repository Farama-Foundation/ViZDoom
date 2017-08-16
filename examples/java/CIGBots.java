import vizdoom.*;

import java.util.*;
import java.lang.*;

public class CIGBots {

    public static void main (String[] args) {

        System.out.println("\n\nCIG BOTS EXAMPLE\n");


        DoomGame game = new DoomGame();

        // Use CIG example config or Your own.
        game.loadConfig("../../scenarios/cig.cfg");

        // Select game and map You want to use.
        game.setDoomGamePath("../../bin/freedoom2.wad");
        //game.setDoomGamePath("../../bin/doom2.wad");   // Not provided with environment due to licences.

        game.setDoomMap("map01");   // Limited deathmatch.
        //game.setDoomMap("map02");   // Full deathmatch.

        // Start multiplayer game only with Your AI (with options that will be used in the competition, details in CIGHost example).
        game.addGameArgs("-host 1 -deathmatch +timelimit 1.0 "
                        + "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1 +sv_nocrouch 1 "
                        + "+viz_respawn_delay 10 +viz_nocheat 1");

        // Name your agent and select color
        // colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
        game.addGameArgs("+name AI +colorset 0");

        game.setMode(Mode.ASYNC_PLAYER);
        game.init();


        int bots = 7;                           // Play with this many bots
        int episodes = 10;                      // Run this many episodes

        for(int i = 0; i < episodes; ++i){

            System.out.println("Episode #" + (i + 1));

            // Add specific number of bots
            // (file examples/bots.cfg must be placed in the same directory as the Doom executable file,
            // edit this file to adjust bots).
            game.sendGameCommand("removebots");
            for(int b = 0; b < bots; ++b) {
                game.sendGameCommand("addbot");
            }

            while(!game.isEpisodeFinished()){   // Play until the game (episode) is over.

                GameState state = game.getState();
                // Analyze the state.

                double[] action = new double[game.getAvailableButtonsSize()];
                // Set your action.

                game.makeAction(action);

                if(game.isPlayerDead()){        // Check if player is dead
                    game.respawnPlayer();       // Use this to respawn immediately after death, new state will be available.
                }

                System.out.println(game.getEpisodeTime() + " Frags: " + game.getGameVariable(GameVariable.FRAGCOUNT));
            }

            System.out.println("Episode finished.");
            System.out.println("************************");
        }

        game.close();
    }
}
