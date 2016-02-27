import enums.*;
import errors.*;
import java.util.*;
import java.lang.Integer;
import java.lang.Boolean;
import java.lang.Thread;
import java.util.ArrayList;
import java.util.List;
import java.lang.InterruptedException;
import java.util.Arrays;
import java.lang.Math;
public class Seed {

	//Example function creating list of all possible moves
	public static List possibilities(int x, long pos){
			List<int[]> list = new ArrayList<int[]>();
			for (long k=0;k<pos;k++){
				int [] action = new int[x];
				for (int j=0;j<x;j++){
				
					if ((k & (long)(1<<(j))) != 0){
						action[j]=1;
					}
					else {
						action[j]=0;
					}
				}
				list.add(action);
				System.out.println(Arrays.toString(action));
			}
			return list;
	}

	public static void main (String[] args) {

		ViziaDoomGameJava game= new ViziaDoomGameJava();
		/*
		Choose the scenario config file you wish to watch.
		Don't load two configs cause the second will overrite the first one.
		Multiple config files are ok but combining these ones doesn't make much sense.

		game.LoadConfig("../config/deadly_corridor.cfg")
		game.LoadConfig("../config/defend_the_center.cfg")
		game.LoadConfig("../config/defend_the_line.cfg")
		game.LoadConfig("../config/health_gathering.cfg")
		game.LoadConfig("../config/my_way_home.cfg")
		game.LoadConfig("../config/predict_position.cfg")
		*/
		game.loadConfig("../config/basic.cfg");
    		game.setDoomEnginePath("../../bin/viziazdoom");
    		game.setDoomGamePath("../../scenarios/doom2.wad");
		game.setScreenResolution(ScreenResolution.RES_640X480);

		int seed = 1234;
		//Sets the seed. It could be after init as well but it's not needed here.
		game.setSeed(seed);
		game.init();

		int actionsNum = game.getAvailableButtonsSize();
		//Number of possible moves
		long pos = (long)Math.pow(2,actionsNum);
		
		List<int[]> actionList = possibilities(actionsNum, pos);
		int episodes = 10;
		long sleepTime = 28;


		for (int i=0;i<episodes;i++){
			int b=i+1;
			System.out.println("Episode #" + b);
			game.newEpisode();
			Random rn = new Random();
			while ( !game.isEpisodeFinished()){
				// Gets the state and possibly to something with it
				State s = game.getState();
				int[] img = s.imageBuffer;
				int[] gameVariables = s.gameVariables;


				//Selecting made action - random one in this example
				int[] selectedAction = actionList.get(rn.nextInt((int)pos));
				// Check which action you chose!
				double reward = game.makeAction(selectedAction);
		
		
				System.out.println("State #" + s.number);
				System.out.println("Game Variables: " + Arrays.toString(gameVariables));
				System.out.println("Made Action: " + Arrays.toString(selectedAction));
				System.out.println("Last Reward: " + reward);
				System.out.println("=====================");

				// Sleep some time because processing is too fast to watch.
				if (sleepTime>0)
				{
					try {
					    Thread.sleep(sleepTime);                 
					} catch(InterruptedException ex) {
					    Thread.currentThread().interrupt();
					}
				}
			}
			System.out.println("Episode finished!");
			System.out.println("Summary reward: " + game.getSummaryReward());
			System.out.println("************************");
		}
		game.close();
	}

}
