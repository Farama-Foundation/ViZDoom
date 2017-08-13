import vizdoom.*;

import java.util.*;
import java.lang.*;

public class DeltaButtons {

    public static void main (String[] args) {

        DoomGame game = new DoomGame();

        System.out.println("\n\nDELTA BUTTONS EXAMPLE\n");
        game.setViZDoomPath("../../bin/vizdoom");

        game.setDoomGamePath("../../bin/freedoom2.wad");
        //game.setDoomGamePath("../../bin/doom2.wad");   // Not provided with environment due to licences.

        game.setDoomMap("map01");

        game.setScreenResolution(ScreenResolution.RES_640X480);

        // Adds delta buttons that will be allowed and set the maximum allowed value (optional).
        game.addAvailableButton(Button.MOVE_FORWARD_BACKWARD_DELTA, 5);
        game.addAvailableButton(Button.MOVE_LEFT_RIGHT_DELTA, 2);
        game.addAvailableButton(Button.TURN_LEFT_RIGHT_DELTA);
        game.addAvailableButton(Button.LOOK_UP_DOWN_DELTA);

        // For normal buttons (binary) all values other than 0 are interpreted as pushed.
        // For delta buttons values determine a precision/speed.
        //
        // For TURN_LEFT_RIGHT_DELTA and LOOK_UP_DOWN_DELTA value is the angle (in degrees)
        // of which the viewing angle will change.
        //
        // For MOVE_FORWARD_BACKWARD_DELTA, MOVE_LEFT_RIGHT_DELTA, MOVE_UP_DOWN_DELTA (rarely used)
        // value is the speed of movement in a given direction (100 is close to the maximum speed).
        List<double[]> actions = new ArrayList<double[]>();
        actions.add(new double[] {10, 1, 1, 1});
        actions.add(new double[] {2, -3, -2, -1});


        // If button's absolute value > max button's value then value = max value with original value sign.

        // Delta buttons in spectator modes correspond to mouse movements.
        // Maximum allowed values also apply to spectator modes.
        // game.addGameArgs("+freelook 1");    //Use this to enable looking around with the mouse.
        // game.setMode(Mode.SPECTATOR);

        game.setWindowVisible(true);
        game.init();

        Random ran = new Random();

        // Run this many episodes.
        int episodes = 10;

        // Use this to remember last shaping reward value.
        double lastTotalShapingReward = 0;

        for (int i = 0; i < episodes; ++i) {

            System.out.println("Episode #" + (i + 1));
            game.newEpisode();

            while (!game.isEpisodeFinished()) {

                // Get the state
                GameState state = game.getState();

                // Make random action and get reward
                game.makeAction(actions.get(ran.nextInt(2)));

            }
        }

        // It will be done automatically in destructor but after close You can init it again with different settings.
        game.close();

    }
}
