import vizdoom.*;

import java.util.*;
import java.lang.*;

public class CIGBots {

    public static void main (String[] args) {

        DoomGame game = new DoomGame();

        System.out.println("\n\nCIG BOTS EXAMPLE\n");

        // Use CIG example config or Your own.
        game.loadConfig("../../examples/config/cig.cfg");

        // Select game and map You want to use.
        game.setDoomGamePath("../../scenarios/freedoom2.wad");
        //game.setDoomGamePath("../../scenarios/doom2.wad");   // Not provided with environment due to licences.

        game.setDoomMap("map01");   // Limited deathmatch.
        //game.setDoomMap("map02");   // Full deathmatch.

        // Start multiplayer game only with Your AI (with options that will be used in the competition, details in CIGHost example).
        game.addGameArgs("-host 1 -deathmatch +timelimit 10.0 +sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1");

        // Name your agent and select color
        // colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
        game.addGameArgs("+name AI +colorset 0");

        game.setMode(Mode.ASYNC_PLAYER);        // Multiplayer requires the use of asynchronous modes.
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

                if(game.isPlayerDead()){        // Check if player is dead
                    game.respawnPlayer();       // Use this to respawn immediately after death, new state will be available.

                    // Or observe the game until automatic respawn.
                    //game.advanceAction();
                    //continue;
                }

                GameState state = game.getState();
                // Analyze the state.

                int[] action= new int[game.getAvailableButtonsSize()];
                // Set your action.

                game.makeAction(action);

                System.out.println(game.getEpisodeTime() + " Frags: " + game.getGameVariable(GameVariable.FRAGCOUNT));
            }

            System.out.println("Episode finished.");
            System.out.println("************************");
        }

        game.close();
    }
}
