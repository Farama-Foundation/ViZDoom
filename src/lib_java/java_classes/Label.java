package vizdoom;

import java.util.ArrayList;
import java.util.List;
import vizdoom.*;

public class Label{
    public int objectId;
    public String objectName;
    public byte value;

    public int x;
    public int y;
    public int width;
    public int height;

    public double objectPositionX;
    public double objectPositionY;
    public double objectPositionZ;
    public double objectAngle;
    public double objectPitch;
    public double objectRoll;
    public double objectVelocityX;
    public double objectVelocityY;
    public double objectVelocityZ;

    Label(int id, String name, byte value, int x, int y, int width, int height,
        double positionX, double positionY, double positionZ,
        double angle, double pitch, double roll,
        double velocityX, double velocityY, double velocityZ){

        this.objectId = objectId;
        this.objectName = objectName;
        this.value = value;

        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;

        this.objectPositionX = positionX;
        this.objectPositionY = positionY;
        this.objectPositionZ = positionZ;

        this.objectAngle = angle;
        this.objectPitch = pitch;
        this.objectRoll = roll;

        this.objectVelocityX = velocityX;
        this.objectVelocityY = velocityY;
        this.objectVelocityZ = velocityZ;
    }
}
