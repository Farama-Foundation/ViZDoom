import vizdoom.*;

import java.util.*;
import java.lang.*;

public class Spectator {

    public static void main (String[] args) {

        DoomGame game= new DoomGame();
        // Choose scenario config file you wish to watch.
        // Don't load two configs cause the second will overwrite the first one.
        // Multiple config files are ok but combining these ones doesn't make much sense.

        //game.loadConfig("../../scenarios/basic.cfg");
        //game.loadConfig("../../scenarios/deadly_corridor.cfg");
        game.loadConfig("../../scenarios/deathmatch.cfg");
        //game.loadConfig("../../scenarios/defend_the_center.cfg");
        //game.loadConfig("../../scenarios/defend_the_line.cfg");
        //game.loadConfig("../../scenarios/health_gathering.cfg");
        //game.loadConfig("../../scenarios/my_way_home.cfg");
        //game.loadConfig("../../scenarios/predict_position.cfg");
        //game.loadConfig("../../scenarios/take_cover.cfg");

        game.setScreenResolution(ScreenResolution.RES_640X480);

        game.setViZDoomPath("../../bin/vizdoom");
        game.setDoomGamePath("../../bin/freedoom2.wad");
        //game.setDoomGamePath("../../bin/doom2.wad");   // Not provided with environment due to licences.

        //Adds mouse support:
        game.addAvailableButton(Button.TURN_LEFT_RIGHT_DELTA);

        // Enables spectator mode, so you can play. Agent is supposed to watch you playing and learn from it.
        game.setWindowVisible(true);
        game.setMode(Mode.SPECTATOR);
        game.init();

        int episodes = 10;
        for (int i = 0; i < episodes; i++){

            System.out.println("Episode #" + (i + 1));

            game.newEpisode();
            while (! game.isEpisodeFinished()){
                GameState state = game.getState();

                game.advanceAction();
                double[] action = game.getLastAction();
                double reward = game.getLastReward();

                System.out.println("State #" + state.number);
                System.out.println("Game Variables: " + Arrays.toString(state.gameVariables));
                System.out.println("Action: " + Arrays.toString(action));
                System.out.println("Reward: " + reward);
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
