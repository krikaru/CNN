import java.util.List;

public interface Layer {
    public void start();
    public void editWeight();
    public List<Neuron[][]> getNeuronList();
}
