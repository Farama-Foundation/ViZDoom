package vizdoom;

import java.util.ArrayList;
import java.util.List;
import vizdoom.*;

public class GameState{
    public int number;
    public int[] gameVariables;

    public int[] screenBuffer;
    public int[] depthBuffer;
    public int[] labelsBuffer;
    public int[] automapBuffer;

    public Label[] labels;

    GameState(int number,
        int[] gameVariables,
        int[] screenBuffer,
        int[] depthBuffer,
        int[] labelsBuffer,
        int[] automapBuffer,
        Label[] labels){

        this.number = number;
        this.gameVariables = gameVariables;
        this.screenBuffer = screenBuffer;
        this.depthBuffer = depthBuffer;
        this.labelsBuffer = labelsBuffer;
        this.automapBuffer = automapBuffer;
        this.labels = labels;
    }
}
