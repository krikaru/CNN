import java.util.List;

public class HiddenLayer {
    List<Neuron> hiddenList;

    private final double E = 0.025;
    private final double A = 0.005;


    public HiddenLayer(List<Neuron> hiddenList) {
        this.hiddenList = hiddenList;
    }

    public void start(){
        for (Neuron neuronhidden: hiddenList){

            double sum = 0;
            for (int i = 0; i < neuronhidden.getPrevWeight().size(); i++){
                sum += neuronhidden.getNeuronPrev().get(i).getValue() * neuronhidden.getPrevWeight().get(i).getValue();
            }
            neuronhidden.setValue(MathForCNN.hyperTan(sum));
        }
    }

    public void editWeight(){
        double delta = 0;
        double gradient = 0;
        double dW = 0;

        for (Neuron hiddenNeuron: hiddenList){
            delta = MathForCNN.derivativeHyperTan(hiddenNeuron.getValue()) *
                    hiddenNeuron.getWeightList().get(0).getValue()*
                    hiddenNeuron.getNeuronNext().get(0).getDelta();
            hiddenNeuron.setDelta(delta);

            gradient = hiddenNeuron.getNeuronNext().get(0).getDelta() * hiddenNeuron.getValue();
            hiddenNeuron.getWeightList().get(0).setGradient(gradient);

            dW = E * gradient + A * hiddenNeuron.getWeightList().get(0).getdW();
            hiddenNeuron.getWeightList().get(0).setdW(dW);

            hiddenNeuron.getWeightList().get(0).setValue(hiddenNeuron.getWeightList().get(0).getValue() + dW);
        }
    }
}
