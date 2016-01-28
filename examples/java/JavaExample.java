
import enums.Button;
import enums.GameVar;
import enums.ScreenFormat;
import enums.Mode;
import enums.ScreenResolution;
import java.util.*;
import java.lang.Integer;
import java.lang.Boolean;
public class JavaExample {

public static void main (String[] args) {
	
	ViziaDoomGameJava dg= new ViziaDoomGameJava();
	System.out.println("VIZIA MAIN EXAMPLE");


    dg.setDoomGamePath("../../bin/viziazdoom");
    dg.setDoomIwadPath("../../scenarios/doom2.wad");
    dg.setDoomFilePath("../../scenarios/s1_b.wad");
    dg.setDoomMap("map01");
    dg.setEpisodeTimeout(200);
    dg.setLivingReward(-1);

    dg.setScreenResolution(ScreenResolution.RES_320X240);
System.out.println("1");
    dg.setRenderHud(false);
    dg.setRenderCrosshair(false);
    dg.setRenderWeapon(true);
    dg.setRenderDecals(false);
    dg.setRenderParticles(false);

    dg.setWindowVisible(true);

    dg.setConsoleEnabled(true);
System.out.println("2");
    dg.addAvailableButton(Button.MOVE_LEFT);
    dg.addAvailableButton(Button.MOVE_RIGHT);
    dg.addAvailableButton(Button.ATTACK);
System.out.println("3");
	GameVar bob = GameVar.HEALTH;
System.out.println("4");
    dg.addAvailableGameVariable(GameVar.HEALTH);
System.out.println("5");
    dg.addAvailableGameVariable(GameVar.KILLCOUNT);


    dg.init();
    //dg->newEpisode();
    int[] action=new int[3];

    action[0] = 1;
    action[1] = 0;
    action[2] = 1;

    int iterations = 100;
    int ep=1;
    for(int i = 0;i<iterations; ++i){

        if( dg.isEpisodeFinished() ){
            dg.newEpisode();
        }
        State s = dg.getState();

        System.out.println( "STATE NUMBER: " + s.number + " HP: " + s.vars[0] + " KILLS: " + s.vars[1] );

        dg.setAction(action);
        //dg.advanceAction();
	dg.makeAction(action)
         System.out.println("reward: "+dg.getLastReward());
    }
    dg.close();
	System.out.println("KONIEC");
  }

}
