package vizdoom;

import java.util.ArrayList;
import java.util.List;
import vizdoom.*;

public class GameState{
    public int id;
    public double[] gameVariables;

    public byte[] screenBuffer;
    public byte[] depthBuffer;
    public byte[] labelsBuffer;
    public byte[] automapBuffer;

    public Label[] labels;

    GameState(int id,
        double[] gameVariables,
        byte[] screenBuffer,
        byte[] depthBuffer,
        byte[] labelsBuffer,
        byte[] automapBuffer,
        Label[] labels){

        this.id = id;
        this.gameVariables = gameVariables;
        this.screenBuffer = screenBuffer;
        this.depthBuffer = depthBuffer;
        this.labelsBuffer = labelsBuffer;
        this.automapBuffer = automapBuffer;
        this.labels = labels;
    }
}
