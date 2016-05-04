import vizdoom.*;

import java.util.*;
import java.lang.*;

public class Multiplayer {

    public static void main (String[] args) {

        DoomGame game = new DoomGame();

        System.out.println("\n\nnMULTIPLAYER EXAMPLE\n");

        game.loadConfig("../../examples/config/multi.cfg");

        // Select game and map You want to use.
        game.setDoomGamePath("../../scenarios/freedoom2.wad");
        //game.setDoomGamePath("../../scenarios/doom2.wad");   // Not provided with environment due to licences.

        // Join existing game (see MultiplayerHost.cpp example)
        game.addGameArgs("-join 127.0.0.1");        // Connect to a host for a multiplayer game.

        game.setMode(Mode.ASYNC_PLAYER);            // Multiplayer requires the use of asynchronous modes.
        game.init();

        List<int[]> actions = new ArrayList<int[]>();
            actions.add(new int[] {1, 0, 1});
        actions.add(new int[] {0, 1, 1});
        actions.add(new int[] {0, 0, 1});

        Random ran = new Random();


        while(!game.isEpisodeFinished()){       // Play until the game (episode) is over.

            if(game.isPlayerDead()){            // Check if player is dead
                game.respawnPlayer();           // Use this to respawn immediately after death, new state will be available.

                // Or observe the game until automatic respawn.
                // game.advanceAction();
                // continue;
            }

            // Get the state
            GameState s = game.getState();

            // Make random action and get reward
            double r = game.makeAction(actions.get(ran.nextInt(3)));

            // You can also get last reward by using this function
            // double r = game.getLastReward();

            System.out.println("State #" + s.number);
            System.out.println("Action reward: " + r);
            System.out.println("Frags: " + game.getGameVariable(GameVariable.FRAGCOUNT));
            System.out.println("=====================");

        }

        game.close();
    }
}
