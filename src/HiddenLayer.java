import java.util.List;

public class HiddenLayer {
    List<Neuron> hiddenList;

    private final double E = 0.1;
    private final double A = 0.003;


    public HiddenLayer(List<Neuron> hiddenList) {
        this.hiddenList = hiddenList;
    }

    public void start(){
//
//                System.out.println("Лист весов между enter and hidden");
//        System.out.println(container.getHiddenList().get(0).getPrevWeight());
        int iter = 0;
        for (Neuron neuronhidden: hiddenList){

            double sum = 0;
            for (int i = 0; i < neuronhidden.getPrevWeight().size(); i++){
                sum += neuronhidden.getNeuronPrev().get(i).getValue() * neuronhidden.getPrevWeight().get(i).getValue();
            }
//            System.out.println("hidden neuron");
//            System.out.println(neuronhidden.getPrevWeight());
            neuronhidden.setValue(MathForCNN.hyperTan(sum));
        }
//        System.out.println("Лист нейронов на hiddden слое: ВЕСЬ");
//        System.out.println(container.getHiddenList());
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

//        System.out.println("----------- hidden.editWeight------------");
//
//        System.out.println("Выход текущ нейрона:");
//        System.out.println(container.getHiddenList().get(0).getValue());
//
//        System.out.println("weight between hidden and out: 0 neuron");
//        System.out.println(container.getHiddenList().get(0).getWeightList());
//        System.out.println("hidden's neuronNext: DELTA");
//        System.out.println(container.getHiddenList().get(0).getNeuronNext());
//
//        System.out.println("Delta hidden neuron : 0 neuron");
//        System.out.println(container.getHiddenList().get(0).getDelta());
//
//        System.out.println("gradient hidden neuron : 0 neuron");
//        System.out.println(container.getHiddenList().get(0).getWeightList().get(0).getGradient());
//
//        System.out.println("dW hidden neuron: 0 neuron");
//        System.out.println(container.getHiddenList().get(0).getWeightList().get(0).getdW());
//
//        System.out.println("New weight between hidden and out: 0 neuron");
//        System.out.println(container.getHiddenList().get(0).getWeightList());

    }
}
