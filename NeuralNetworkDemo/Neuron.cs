namespace NeuralNetworkDemo;

/// <summary>
/// Rappresenta un singolo neurone con attivazione sigmoide.
/// </summary>
public class Neuron
{
    public double[] Weights { get; private set; }
    public double Bias { get; private set; }
    public double Output { get; private set; }
    public double Delta { get; set; }

    private readonly Random _random;

    public Neuron(int numberOfInputs, Random random)
    {
        _random = random;
        Weights = new double[numberOfInputs];
        for (int i = 0; i < numberOfInputs; i++)
        {
            Weights[i] = _random.NextDouble() * 2 - 1; // valori tra -1 e 1
        }
        Bias = _random.NextDouble() * 2 - 1;
    }

    /// <summary>
    /// Funzione di attivazione sigmoide: f(x) = 1 / (1 + e^(-x))
    /// </summary>
    public static double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }

    /// <summary>
    /// Derivata della sigmoide: f'(x) = f(x) * (1 - f(x))
    /// </summary>
    public static double SigmoidDerivative(double sigmoidOutput)
    {
        return sigmoidOutput * (1.0 - sigmoidOutput);
    }

    /// <summary>
    /// Calcola l'output del neurone: sigmoide(somma pesata + bias)
    /// </summary>
    public double Forward(double[] inputs)
    {
        double sum = Bias;
        for (int i = 0; i < Weights.Length; i++)
        {
            sum += inputs[i] * Weights[i];
        }
        Output = Sigmoid(sum);
        return Output;
    }

    /// <summary>
    /// Aggiorna i pesi e il bias usando il gradiente discendente.
    /// </summary>
    public void UpdateWeights(double[] inputs, double learningRate)
    {
        for (int i = 0; i < Weights.Length; i++)
        {
            Weights[i] += learningRate * Delta * inputs[i];
        }
        Bias += learningRate * Delta;
    }
}
