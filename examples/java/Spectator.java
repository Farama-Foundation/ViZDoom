import vizdoom.*;

import java.util.*;
import java.lang.*;

public class Spectator {

    public static void main (String[] args) {

        DoomGame game= new DoomGame();
        // Choose scenario config file you wish to watch.
        // Don't load two configs cause the second will overwrite the first one.
        // Multiple config files are ok but combining these ones doesn't make much sense.

        //game.loadConfig("../../examples/config/basic.cfg");
        //game.loadConfig("../../examples/config/deadly_corridor.cfg");
        game.loadConfig("../../examples/config/deathmatch.cfg");
        //game.loadConfig("../../examples/config/defend_the_center.cfg");
        //game.loadConfig("../../examples/config/defend_the_line.cfg");
        //game.loadConfig("../../examples/config/health_gathering.cfg");
        //game.loadConfig("../../examples/config/my_way_home.cfg");
        //game.loadConfig("../../examples/config/predict_position.cfg");
        //game.loadConfig("../../examples/config/take_cover.cfg");

        game.setScreenResolution(ScreenResolution.RES_640X480);

        // Select game and map You want to use.
        game.setDoomGamePath("../../scenarios/freedoom2.wad");

        //game.setDoomGamePath("../../scenarios/doom2.wad");
        game.setViZDoomPath("../../bin/vizdoom");

        //Adds mouse support:
        game.addAvailableButton(Button.TURN_LEFT_RIGHT_DELTA);

        // Enables spectator mode, so you can play. Agent is supposed to watch you playing and learn from it.
        game.setWindowVisible(true);
        game.setMode(Mode.SPECTATOR);
        game.init();

        int episodes = 10;
        for (int i=0;i<episodes;i++){

            System.out.println("Episode #" +(i+1));

            game.newEpisode();
            while (! game.isEpisodeFinished()){
                GameState s = game.getState();
                int[] img = s.imageBuffer;
                int[] misc = s.gameVariables;

                game.advanceAction();
                boolean[] a = game.getLastAction();
                double r = game.getLastReward();

                System.out.println("State #"+s.number);
                System.out.println("Game Variables: "+Arrays.toString(misc));
                System.out.println("Action: "+ Arrays.toString(a));
                System.out.println("Reward: "+r);
                System.out.println("=====================");
            }

            System.out.println("episode finished!");
            System.out.println("Total reward:"+ game.getTotalReward());
            System.out.println("************************");

            try {
                Thread.sleep(2000);
            }
            catch(InterruptedException ex){
                Thread.currentThread().interrupt();
            }
        }
        game.close();
    }
}
