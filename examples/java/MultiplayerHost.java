import vizdoom.*;

import java.util.*;
import java.lang.*;

public class MultiplayerHost {

    public static void main (String[] args) {

        DoomGame game = new DoomGame();

        System.out.println("\n\nnMULTIPLAYER HOST EXAMPLE\n");

        game.loadConfig("../../examples/config/multi.cfg");

        // Select game and map You want to use.
        game.setDoomGamePath("../../scenarios/freedoom2.wad");
        //game.setDoomGamePath("../../scenarios/doom2.wad");   // Not provided with environment due to licences.

        // Host game.
        game.addGameArgs("-host 2 -deathmatch +map map01");

        game.setMode(Mode.ASYNC_SPECTATOR);         // Multiplayer requires the use of asynchronous modes.
        game.init();

        while(!game.isEpisodeFinished()){           // Play until the game (episode) is over.

            if(game.isPlayerDead()){                // Check if player is dead
                game.respawnPlayer();               // Use this to respawn immediately after death, new state will be available.

                // Or observe the game until automatic respawn.
                // game.advanceAction();
                // continue;
            }

            game.advanceAction();
        }

        game.close();
    }
}
