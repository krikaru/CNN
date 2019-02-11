import java.util.Arrays;
import java.util.List;

public class ConvoluationalLayer {
    Container container;
    List<Neuron[][]> convList;
    List<Weight[][]> kernelsList;
    List<Neuron[][]> splitList;

    public ConvoluationalLayer(Container container) {
        this.container = container;
    }

    public void start(){
        convList = container.getConvList();
        kernelsList = container.getBetweenSplitAndConv();
        splitList = container.getSplitList();

        Neuron prevNeuron;
        Weight[][] kernel;
        Weight weight;
        Neuron thisNeuron;


//        System.out.println("Массив ядра весов от split до conv(0 ядро):");
//        System.out.println(Arrays.deepToString(kernelsList.get(0)));


        for (int count = 0; count < 6; count++){

            kernel = kernelsList.get(count);

            for (int columnNew = 0; columnNew < splitList.get(count)[0].length - kernel[0].length + 1; columnNew++){
                for (int rowNew = 0; rowNew < splitList.get(count).length - kernel.length + 1 ; rowNew++){
                    double sum = 0;
                    int kernelCount = -1;
                    thisNeuron = convList.get(count)[rowNew][columnNew];

                    for (int columnK = 0; columnK < kernel[0].length; columnK++){
                        for (int rowK = 0; rowK < kernel.length; rowK++){

                            prevNeuron = splitList.get(count)[rowK + rowNew][columnK + columnNew];
                            weight = kernelsList.get(count)[rowK][columnK];
                            kernelCount++;

                            sum += weight.getValue()* prevNeuron.getValue();

                            //set connection between this and previosly neuron
//                            System.err.println(prevNeuron.getNeuronNext());;
                            prevNeuron.getNeuronNext().set(kernelCount, thisNeuron);
                            prevNeuron.getWeightList().set(kernelCount, weight);

                            thisNeuron.getNeuronPrev().set(kernelCount, prevNeuron);
                            thisNeuron.getPrevWeight().set(kernelCount, weight);
                        }
                    }
                    thisNeuron.setValue(MathForCNN.leakyReLU(sum));
                }
            }
        }

//        System.out.println("Conv:");
//        System.out.println(Arrays.deepToString(convList.get(0)));
    }

    public void editWeight(){
//        System.out.println("------conv.editWeight--------");
//        System.out.println(Arrays.deepToString(container.getConvList().get(0)));

    }
}
