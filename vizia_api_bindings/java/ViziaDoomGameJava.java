import java.util.ArrayList;
import java.util.List;
//import errors.*;
import enums.*;
public class ViziaDoomGameJava{
	static {
      		System.loadLibrary("DoomLIB"); 
   	}

	public long internalPtr = 0; 
	public ViziaDoomGameJava(){ 
		DoomGame();
	}

	public native int DoomTics2Ms(int tics);
 	public native int Ms2DoomTics(int ms);
	public native float DoomFixedToFloat(int doomFixed);
//----------------------------------------------------------------------
	private native void DoomGame(); 
	private native void DoomGameDes();
//----------------------------------------------------------------------
	public native boolean loadConfig(String file); 
	public native boolean saveConfig(String file); 	
	
	public native boolean init(); 
	public native void close();

	public native void newEpisode();
	public native boolean isRunning(); 
	
	public native void setNextAction(int[] actions);
        public native void advanceAction();
        public native void advanceAction(boolean stateUpdate, boolean renderOnly);
        public native void advanceAction(boolean stateUpdate, boolean renderOnly, int tics);
	public native State getState(); 
	public native boolean[] getLastAction();
	public native boolean isNewEpisode();
	public native boolean isEpisodeFinished();
	public native void addAvailableButton(Button button); 
	public native void addAvailableButton(Button button, int maxValue); 	
        public native void clearAvailableButtons();
        public native int getAvailableButtonsSize();
        public native void setButtonMaxValue(Button button, int maxValue);
	public native void addStateAvailableVar(GameVar var); 
	public native void clearStateAvailableVars();
        public native int getStateAvailableVarsSize();
 	public native void addCustomGameArg(String arg);
        public native void clearCustomGameArgs();

        public native void sendGameCommand(String cmd);	
	public native int[] getGameScreen();
	
	private native int getGameMod(); 		
	public GameMode getGameMode(){
		GameMode ret=GameMode.values()[getGameMod()];
		return ret;

	}
	
	public native void setGameMode(GameMode mode);

	public native int getGameVar(GameVar var);

        public native float getLivingReward();
       	public native  void setLivingReward(float livingReward);
        public native float getDeathPenalty();
        public native void setDeathPenalty(float deathPenalty);

        public native float getLastReward();
        public native float getSummaryReward();

        public native void setDoomGamePath(String path);
        public native void setDoomIwadPath(String path);
        public native void setDoomFilePath(String path);
        public native void setDoomMap(String map);
        public native void setDoomSkill(int skill);
        public native void setDoomConfigPath(String path);

        public native int getSeed();
        public native void setSeed(int seed);

        public native void setAutoNewEpisode(boolean set);
        public native void setNewEpisodeOnTimeout(boolean set);
        public native void setNewEpisodeOnPlayerDeath(boolean set);
        public native void setNewEpisodeOnMapEnd(boolean set);

        public native int getEpisodeStartTime();
        public native void setEpisodeStartTime(int tics);

        public native int getEpisodeTimeout();
        public native void setEpisodeTimeout(int tics);

        public native void setScreenResolution(int width, int height);
        public native void setScreenWidth(int width);
        public native void setScreenHeight(int height);
        public native void setScreenFormat(ScreenFormat format);
        public native void setRenderHud(boolean hud);
        public native void setRenderWeapon(boolean weapon);
        public native void setRenderCrosshair(boolean crosshair);
        public native void setRenderDecals(boolean decals);
        public native void setRenderParticles(boolean particles);
        public native void setVisibleWindow(boolean visibility);
        public native void setDisabledConsole(boolean noConsole);

        public native int getScreenWidth();
        public native int getScreenHeight();
        public native int getScreenChannels();
        public native int getScreenPitch();
        public native int getScreenSize();
	private native int getScreenForma();
        public  ScreenFormat getScreenFormat(){
		ScreenFormat ret=ScreenFormat.values()[getScreenForma()];
		return ret;
	}
		
}
