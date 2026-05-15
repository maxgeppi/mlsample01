namespace NeuralNetworkDemo;

/// <summary>
/// Rete neurale semplice con uno strato nascosto per classificazione binaria.
/// Architettura: 2 input -> N neuroni nascosti -> 1 neurone di output
/// </summary>
public class NeuralNetwork
{
    public Neuron[] HiddenLayer { get; }
    public Neuron OutputNeuron { get; }
    public double LearningRate { get; }

    /// <summary>
    /// Crea una rete neurale con il numero specificato di input e neuroni nascosti.
    /// </summary>
    public NeuralNetwork(int numberOfInputs, int hiddenNeurons, double learningRate, int? seed = null)
    {
        LearningRate = learningRate;
        var random = seed.HasValue ? new Random(seed.Value) : new Random();

        HiddenLayer = new Neuron[hiddenNeurons];
        for (int i = 0; i < hiddenNeurons; i++)
        {
            HiddenLayer[i] = new Neuron(numberOfInputs, random);
        }

        OutputNeuron = new Neuron(hiddenNeurons, random);
    }

    /// <summary>
    /// Propagazione in avanti: calcola l'output della rete.
    /// </summary>
    public double Forward(double[] inputs)
    {
        var hiddenOutputs = new double[HiddenLayer.Length];
        for (int i = 0; i < HiddenLayer.Length; i++)
        {
            hiddenOutputs[i] = HiddenLayer[i].Forward(inputs);
        }

        return OutputNeuron.Forward(hiddenOutputs);
    }

    /// <summary>
    /// Addestramento con backpropagation per un singolo esempio.
    /// </summary>
    public void Train(double[] inputs, double expected)
    {
        double predicted = Forward(inputs);

        // Errore dello strato di output
        double outputError = expected - predicted;
        OutputNeuron.Delta = outputError * Neuron.SigmoidDerivative(predicted);

        // Errore dello strato nascosto (backpropagation)
        for (int i = 0; i < HiddenLayer.Length; i++)
        {
            double hiddenError = OutputNeuron.Delta * OutputNeuron.Weights[i];
            HiddenLayer[i].Delta = hiddenError * Neuron.SigmoidDerivative(HiddenLayer[i].Output);
        }

        // Aggiornamento pesi dello strato di output
        var hiddenOutputs = new double[HiddenLayer.Length];
        for (int i = 0; i < HiddenLayer.Length; i++)
        {
            hiddenOutputs[i] = HiddenLayer[i].Output;
        }
        OutputNeuron.UpdateWeights(hiddenOutputs, LearningRate);

        // Aggiornamento pesi dello strato nascosto
        for (int i = 0; i < HiddenLayer.Length; i++)
        {
            HiddenLayer[i].UpdateWeights(inputs, LearningRate);
        }
    }

    /// <summary>
    /// Addestramento della rete per un numero specificato di epoche.
    /// </summary>
    public void Train(double[][] allInputs, double[] allExpected, int epochs)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalError = 0;
            for (int i = 0; i < allInputs.Length; i++)
            {
                Train(allInputs[i], allExpected[i]);
                double error = allExpected[i] - Forward(allInputs[i]);
                totalError += error * error;
            }

            if ((epoch + 1) % 1000 == 0)
            {
                Console.WriteLine($"  Epoca {epoch + 1}/{epochs} - Errore totale: {totalError:F6}");
            }
        }
    }

    /// <summary>
    /// Predice la classe (0 o 1) per un dato input.
    /// </summary>
    public int Predict(double[] inputs)
    {
        double output = Forward(inputs);
        return output >= 0.5 ? 1 : 0;
    }
}
