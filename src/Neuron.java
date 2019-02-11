import java.util.ArrayList;
import java.util.List;

public class Neuron {

    private double value;
    private double delta = 0;
    private List<Neuron> neuronNext = new ArrayList<Neuron>();
    private List<Weight> weightList = new ArrayList<Weight>();
    private List<Neuron> neuronPrev = new ArrayList<Neuron>();
    private List<Weight> prevWeight = new ArrayList<Weight>();
    private int rowMax = 0;
    private int columnMax = 0;

    public Neuron(double value) {
        this.value = value;
    }

    public int getRowMax() {
        return rowMax;
    }

    public void setRowMax(int rowMax) {
        this.rowMax = rowMax;
    }

    public int getColumnMax() {
        return columnMax;
    }

    public void setColumnMax(int columnMax) {
        this.columnMax = columnMax;
    }

    public List<Neuron> getNeuronPrev() {
        return neuronPrev;
    }

    public void setNeuronPrev(List<Neuron> neuronPrev) {
        this.neuronPrev = neuronPrev;
    }

    public List<Weight> getPrevWeight() {
        return prevWeight;
    }

    public void setPrevWeight(List<Weight> prevWeight) {
        this.prevWeight = prevWeight;
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }

    public double getDelta() {
        return delta;
    }

    public void setDelta(double delta) {
        this.delta = delta;
    }

    public List<Neuron> getNeuronNext() {
        return neuronNext;
    }

    public void setNeuronNext(List<Neuron> neuronNext) {
        this.neuronNext = neuronNext;
    }

    public List<Weight> getWeightList() {
        return weightList;
    }

    public void setWeightList(List<Weight> weightList) {
        this.weightList = weightList;
    }

    @Override
    public String toString() {
        return "Neuron{" +
                "value=" + value +
                ", delta=" + delta +
                '}';
    }
}
