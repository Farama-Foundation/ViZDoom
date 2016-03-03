/*
#####################################################################
# This script presents SPECTATOR mode. In SPECTATOR mode you play and
# your agent can learn from it.
# gguration is loaded from "../config/<SCENARIO_NAME>.cfg" file.
# 
# To see the scenario description go to "../../scenarios/README"
# 
#####################################################################
*/
import enums.*;
import errors.*;
import java.util.*;
import java.lang.Integer;
import java.lang.Boolean;
import java.lang.InterruptedException;
import java.util.Arrays;
public class Spectator {

	public static void main (String[] args) {
	
		ViziaDoomGameJava game= new ViziaDoomGameJava();
		// Choose scenario config file you wish to watch.
		// Don't load two configs cause the second will overrite the first one.
		// Multiple config files are ok but combining these ones doesn't make much sense.

		//game.loadConfig("../config/basic.cfg");
		//game.loadConfig("../config/deadly_corridor.cfg");
		game.loadConfig("../config/deathmatch.cfg");
		//game.loadConfig("../config/defend_the_center.cfg");
		//game.loadConfig("../config/defend_the_line.cfg");
		//game.loadConfig("../config/health_gathering.cfg");
		//game.loadConfig("../config/my_way_home.cfg");
		//game.loadConfig("../config/predict_position.cfg");
		//game.loadConfig("../config/take_cover.cfg");
		game.setScreenResolution(ScreenResolution.RES_640X480);
		game.setDoomGamePath("../../scenarios/doom2.wad");
		game.setDoomEnginePath("../../bin/viziazdoom");
		//Adds mouse support:
		game.addAvailableButton(Button.TURN_LEFT_RIGHT_DELTA);
		// Enables spectator mode, so you can play. Agent is supposed to watch you playing and learn from it.
		game.setWindowVisible(true);
		game.setMode(Mode.SPECTATOR);
		game.init();

		int episodes = 10;
		System.out.println("");
		for (int i=0;i<episodes;i++){
			int b=i+1;
			System.out.println("Episode #" +b);
	
			game.newEpisode();
			while (! game.isEpisodeFinished()){
				GameState s = game.getState();
				int[] img = s.imageBuffer;
				int[] misc = s.gameVariables;

				game.advanceAction();
				boolean[] a = game.getLastAction();
				double r = game.getLastReward();
		
				System.out.println("state #"+s.number);
				System.out.println("game variables: "+Arrays.toString(misc));
				System.out.println("action: "+ Arrays.toString(a));
				System.out.println("reward: "+r);
				System.out.println("=====================");
			}
	
			System.out.println("episode finished!");
			System.out.println("summary reward:"+ game.getSummaryReward());
			System.out.println("************************");
			
			try {
				Thread.sleep(2000);                 
			} catch(InterruptedException ex) {
				Thread.currentThread().interrupt();
			}
		}
		game.close();
	}
}
