import java.util.List;

public class EnterMLP {
     List<Neuron> enterList;
    private List<Neuron[][]> prevList;
    private final double E = 0.00007;
    private final double A = 0.00003;


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

//        System.out.println("весы между subs and enter");
//        System.out.println(enterNeurons.get(0).getPrevWeight());
//        System.out.println("лист нейронов subs");
////
////
////
//        System.out.println(enterNeurons.get(0).getNeuronPrev());
//        System.out.println("Лист нейронов EnterMLP: ВЕСЬ");
//        System.out.println(enterNeurons);
    }

    public void editWeight(){
        double sum = 0;
        double grad = 0;
        double dW = 0;

        for (Neuron enterNeuron: enterList){
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

//        System.out.println("----------- enter.editWeight------------");
//
//        System.out.println("Выход текущ нейрона:");
//        System.out.println(container.getEnterMLPList().get(0).getValue());
//
//        System.out.println("weight between enter and hidden: 0 neuron");
//        System.out.println(container.getEnterMLPList().get(0).getWeightList());
//        System.out.println("hidden's neuronNext: DELTA");
//        System.out.println(container.getEnterMLPList().get(0).getNeuronNext());
////
//        System.out.println("Delta enter neuron : 0 neuron");
//        System.out.println(container.getEnterMLPList().get(0).getDelta());
////
//        System.out.println("gradient enter neuron : 0 neuron");
//        System.out.println(container.getEnterMLPList().get(0).getWeightList().get(0).getGradient());
////
//        System.out.println("dW enter neuron: 0 neuron");
//        System.out.println(container.getEnterMLPList().get(0).getWeightList().get(0).getdW());
////
//        System.out.println("New weight between hidden and out: 0 neuron");
//        System.out.println(container.getEnterMLPList().get(0).getWeightList());
//        for (int i = 0; i < 6; i++){
//            double sum = 0;
//            double grad = 0;
//            double dW = 0;
//            for (int countHidden = 0; countHidden < 12; countHidden++){
//
//                sum += neuronList.get(i).getNeuronNext().get(countHidden).getDelta() * neuronList.get(i).getWeightList().get(countHidden).getValue();
//
//                grad = neuronList.get(i).getNeuronNext().get(countHidden).getDelta() *neuronList.get(i).getValue();
//                neuronList.get(i).getWeightList().get(countHidden).setGradient(grad);
//
//                dW = E * grad + A * neuronList.get(i).getWeightList().get(countHidden).getdW();
//                neuronList.get(i).getWeightList().get(countHidden).setdW(dW);
//
//                neuronList.get(i).getWeightList().get(countHidden).setValue(neuronList.get(i).getWeightList().get(countHidden).getValue() + dW);
//            }
//            neuronList.get(i).setDelta(MathForCNN.derivativeHyperTan(neuronList.get(i).getValue()) * sum);
//
//        }
    }

}
