package vizdoom;

import java.util.ArrayList;
import java.util.List;
import vizdoom.*;

public class Label{
    public int objectId;
    public String objectName;
    public byte value;

    Label(int objectId, String objectName, byte value){
        this.objectId = objectId;
        this.objectName = objectName;
        this.value = value;
    }
}
