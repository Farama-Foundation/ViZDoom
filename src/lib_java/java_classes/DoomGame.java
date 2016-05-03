package vizdoom;

import java.util.ArrayList;
import java.util.List;
import vizdoom.*;

public class DoomGame{
    static {
        System.loadLibrary("vizdoom");
    }

    public long internalPtr = 0;
    public DoomGame(){
        DoomGame();
    }

    public native int DoomTics2Ms(int tics);
    public native int Ms2DoomTics(int ms);
    public native double DoomFixedToDouble(int doomFixed);
    public native boolean isBinaryButton(Button button);
    public native boolean isDeltaButton(Button button);

    private native void DoomGame();
    public native boolean loadConfig(String file);

    public native boolean init();
    public native void close();

    public native void newEpisode();
    public native boolean isRunning();

    public native void setAction(int[] actions);
    public native void advanceAction();
    public native void advanceAction(int tics);
    public native void advanceAction(int tics, boolean stateUpdate, boolean renderOnly);
    public native double makeAction(int[] actions);
    public native double makeAction(int[] actions, int tics);


    public native GameState getState();

    public native boolean[] getLastAction();

    public native boolean isNewEpisode();
    public native boolean isEpisodeFinished();

    public native boolean isPlayerDead();
    public native void respawnPlayer();

    public native void addAvailableButton(Button button);
    public native void addAvailableButton(Button button, int maxValue);
    public native void clearAvailableButtons();
    public native int getAvailableButtonsSize();
    public native void setButtonMaxValue(Button button, int maxValue);
    public native int getButtonMaxValue(Button button);

    public native void addAvailableGameVariable(GameVariable var);

    public native void clearAvailableGameVariables();
    public native int getAvailableGameVariablesSize();

    public native void addGameArgs(String arg);
    public native void clearGameArgs();

    public native void sendGameCommand(String cmd);

    public native int[] getGameScreen();

    private native int getMod();

    public Mode getMode(){
        Mode ret=Mode.values()[getMod()];
        return ret;

    }

    public native void setMode(Mode mode);

    public native int getGameVariable(GameVariable var);

    public native double getLivingReward();
    public native  void setLivingReward(double livingReward);
    public native double getDeathPenalty();
    public native void setDeathPenalty(double deathPenalty);

    public native double getLastReward();
    public native double getTotalReward();

    public native void setViZDoomPath(String path);
    public native void setDoomGamePath(String path);
    public native void setDoomScenarioPath(String path);
    public native void setDoomMap(String map);
    public native void setDoomSkill(int skill);
    public native void setDoomConfigPath(String path);

    public native int getSeed();
    public native void setSeed(int seed);

    public native int getEpisodeStartTime();
    public native void setEpisodeStartTime(int tics);

    public native int getEpisodeTimeout();
    public native void setEpisodeTimeout(int tics);

    public native int getEpisodeTime();

    public native void setScreenResolution(ScreenResolution resolution);
    public native void setScreenFormat(ScreenFormat format);
    public native void setRenderHud(boolean hud);
    public native void setRenderWeapon(boolean weapon);
    public native void setRenderCrosshair(boolean crosshair);
    public native void setRenderDecals(boolean decals);
    public native void setRenderParticles(boolean particles);
    public native void setWindowVisible(boolean visibility);
    public native void setConsoleEnabled(boolean console);
    public native void setSoundEnabled(boolean sound);

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
