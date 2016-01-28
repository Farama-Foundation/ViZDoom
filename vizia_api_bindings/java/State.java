import java.util.ArrayList;
import java.util.List;

public class State{
	public int number;
	public int[] vars;
	public int[] imageBuffer;
	public State(){
		this.number=-1;
		this.vars=null;
		this.imageBuffer=null;
	};
	public State(int number, int[] vars , int[] imageBuffer){
		this.number=number;
		this.vars=vars;
		this.imageBuffer=imageBuffer;
	}
}
