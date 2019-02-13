import java.util.List;

public class EnterMLP {
     List<Neuron> enterList;
    private List<Neuron[][]> prevList;
    private final double E = 0.025;
    private final double A = 0.05;


    public EnterMLP(List<Neuron> enterList, List<Neuron[][]> prevList) {
        this.enterList = enterList;
        this.prevList = prevList;
    }

    public void start() {
        for (int i = 0; i < 6; i++){
            Neuron prevNeuron;
            Neuron thisNeuron;
            double sum = 0;
            int count = -1;
            thisNeuron = enterList.get(i);
            for (int rowC = 0; rowC < prevList.get(i)[0].length; rowC++){
                for (int columnC = 0; columnC < prevList.get(i).length; columnC++){
                    prevNeuron = prevList.get(i)[rowC][columnC];
                    count++;
                    sum += prevNeuron.getValue() * thisNeuron.getPrevWeight().get(count).getValue();

                    //set new value of neuron in this layer
                    thisNeuron.getNeuronPrev().set(count, prevNeuron);

                    //set weight and neuron for previosly neurons
                    prevNeuron.getNeuronNext().set(0, thisNeuron);
                    prevNeuron.getWeightList().set(0, thisNeuron.getPrevWeight().get(count));
                }
            }
            enterList.get(i).setValue(MathForCNN.hyperTan(sum));
        }
    }

    public void editWeight(){
        double grad;
        double dW;

        for (Neuron enterNeuron: enterList){
            double sum = 0;
            for (int countWeight = 0; countWeight < 12; countWeight++){
                sum += enterNeuron.getNeuronNext().get(countWeight).getDelta() * enterNeuron.getWeightList().get(countWeight).getValue();

                grad = enterNeuron.getNeuronNext().get(countWeight).getDelta() * enterNeuron.getValue();
                enterNeuron.getWeightList().get(countWeight).setGradient(grad);

                dW = E * grad + A * enterNeuron.getWeightList().get(countWeight).getdW();
                enterNeuron.getWeightList().get(countWeight).setdW(dW);

                enterNeuron.getWeightList().get(countWeight).setValue(enterNeuron.getWeightList().get(countWeight).getValue() + dW);
            }

            enterNeuron.setDelta(MathForCNN.derivativeHyperTan(enterNeuron.getValue()) * sum);
        }
    }

}
