import java.util.Arrays;
import java.util.List;

public class ConvoluationalLayer implements Layer {

    List<Neuron[][]> convList;
    List<Weight[][]> kernelsList;
    List<Neuron[][]> prevList;

    public ConvoluationalLayer(List<Neuron[][]> convList, List<Weight[][]> kernelList, List<Neuron[][]> prevList) {
        this.convList = convList;
        this.kernelsList = kernelList;
        this.prevList = prevList;
    }

    public void start(){
        Neuron prevNeuron;
        Weight[][] kernel;
        Weight weight;
        Neuron thisNeuron;


//        System.out.println("Массив ядра весов от split до conv(0 ядро):");
//        System.out.println(Arrays.deepToString(kernelsList.get(0)));

        for (int count = 0; count < 6; count++){

            kernel = kernelsList.get(count);


            for (int columnNew = 0; columnNew < prevList.get(count)[0].length - 5 + 1; columnNew++){
                for (int rowNew = 0; rowNew < prevList.get(count).length - 5 + 1 ; rowNew++){
                    double sum = 0;
                    int kernelCount = -1;
                    thisNeuron = convList.get(count)[rowNew][columnNew];

                    for (int columnK = 0; columnK < kernel[0].length; columnK++){
                        for (int rowK = 0; rowK < kernel.length; rowK++){

                            prevNeuron = prevList.get(count)[rowK + rowNew][columnK + columnNew];
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

    @Override
    public void editWeight(){
//        System.out.println("------conv.editWeight--------");
//        System.out.println(Arrays.deepToString(container.getConvList().get(0)));

    }

    @Override
    public List<Neuron[][]> getNeuronList() {
        return convList;
    }
}
