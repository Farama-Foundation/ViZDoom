import vizdoom.*;

import java.util.*;
import java.lang.*;

public class CIG {

    public static void main (String[] args) {

        DoomGame game = new DoomGame();

        System.out.println("\n\nCIG EXAMPLE\n");


        // Use CIG example config or Your own.
        game.loadConfig("../../examples/config/cig.cfg");

        // Select game and map You want to use.
        game.setDoomGamePath("../../scenarios/freedoom2.wad");
        //game.setDoomGamePath("../../scenarios/doom2.wad");   // Not provided with environment due to licences.

        game.setDoomMap("map01");   // Limited deathmatch.
        //game.setDoomMap("map02");   // Full deathmatch.

        // Join existing game.
        game.addGameArgs("-join 127.0.0.1");    // Connect to a host for a multiplayer game.

        // Name Your AI.
        game.addGameArgs("+name AI");

        game.setMode(Mode.ASYNC_PLAYER);        // Multiplayer requires the use of asynchronous modes.
        game.init();

        while(!game.isEpisodeFinished()){       // Play until the game (episode) is over.

            if(game.isPlayerDead()){            // Check if player is dead
                game.respawnPlayer();           // Use this to respawn immediately after death, new state will be available.

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

        game.close();
    }
}
