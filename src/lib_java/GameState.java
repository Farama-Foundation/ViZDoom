package vizdoom;

import java.util.ArrayList;
import java.util.List;

public class GameState{
    public int number;
    public int[] gameVariables;
    public int[] imageBuffer;
    public GameState(){
        this.number=-1;
        this.gameVariables=null;
        this.imageBuffer=null;
    };
    public GameState(int number, int[] vars , int[] imageBuffer){
        this.number=number;
        this.gameVariables=vars;
        this.imageBuffer=imageBuffer;
    }
}
