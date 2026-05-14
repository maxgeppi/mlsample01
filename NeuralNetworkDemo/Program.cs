using NeuralNetworkDemo;

// ============================================================
// Esempio di rete neurale per classificare due tipi di oggetti
// basandosi su due caratteristiche (proprieta).
//
// Scenario: classificare frutti in MELE (0) e ARANCE (1)
//   Caratteristica 1: Peso (grammi, normalizzato)
//   Caratteristica 2: Colore (scala 0-1, 0=verde, 1=arancione)
//
// Architettura: 2 input -> 3 neuroni nascosti -> 1 output
// Attivazione: Sigmoide
// ============================================================

Console.WriteLine("=== Rete Neurale con Sigmoide in C# ===");
Console.WriteLine();
Console.WriteLine("Classificazione di frutti: MELA (0) vs ARANCIA (1)");
Console.WriteLine("Caratteristiche: [Peso normalizzato, Colore (0=verde, 1=arancione)]");
Console.WriteLine();

// Dati di addestramento: [peso_normalizzato, colore]
// Mele: leggere e verdi -> classe 0
// Arance: pesanti e arancioni -> classe 1
double[][] trainingInputs =
[
    [0.1, 0.1],  // Mela piccola, verde
    [0.2, 0.2],  // Mela media, verdina
    [0.15, 0.0], // Mela piccola, molto verde
    [0.3, 0.15], // Mela grande, verdina
    [0.25, 0.1], // Mela media, verde
    [0.8, 0.9],  // Arancia grande, arancione
    [0.7, 0.85], // Arancia media, arancione
    [0.9, 1.0],  // Arancia molto grande, molto arancione
    [0.75, 0.8], // Arancia media, arancione
    [0.85, 0.95] // Arancia grande, molto arancione
];

double[] expectedOutputs =
[
    0, 0, 0, 0, 0,  // Mele
    1, 1, 1, 1, 1   // Arance
];

// Creazione della rete neurale
// 2 input, 3 neuroni nascosti, learning rate 0.5, seed per riproducibilita
var network = new NeuralNetwork(
    numberOfInputs: 2,
    hiddenNeurons: 3,
    learningRate: 0.5,
    seed: 42
);

// Addestramento
Console.WriteLine("--- Fase di Addestramento (5000 epoche) ---");
Console.WriteLine();
network.Train(trainingInputs, expectedOutputs, epochs: 5000);

// Verifica sui dati di addestramento
Console.WriteLine();
Console.WriteLine("--- Risultati sui dati di addestramento ---");
Console.WriteLine();
Console.WriteLine($"{"Input",-20} {"Atteso",-10} {"Predetto",-10} {"Output Raw",-12}");
Console.WriteLine(new string('-', 52));

for (int i = 0; i < trainingInputs.Length; i++)
{
    double raw = network.Forward(trainingInputs[i]);
    int predicted = network.Predict(trainingInputs[i]);
    string label = predicted == 0 ? "MELA" : "ARANCIA";
    Console.WriteLine(
        $"[{trainingInputs[i][0]:F2}, {trainingInputs[i][1]:F2}]     " +
        $"{expectedOutputs[i],-10}" +
        $"{label,-10}" +
        $"{raw:F6}");
}

// Test con nuovi dati
Console.WriteLine();
Console.WriteLine("--- Predizioni su nuovi dati ---");
Console.WriteLine();

double[][] testInputs =
[
    [0.1, 0.05],  // Dovrebbe essere MELA
    [0.35, 0.3],  // Dovrebbe essere MELA
    [0.6, 0.7],   // Dovrebbe essere ARANCIA
    [0.95, 0.9],  // Dovrebbe essere ARANCIA
    [0.5, 0.5]    // Caso ambiguo
];

foreach (var input in testInputs)
{
    double raw = network.Forward(input);
    int predicted = network.Predict(input);
    string label = predicted == 0 ? "MELA" : "ARANCIA";
    Console.WriteLine($"  Input: [{input[0]:F2}, {input[1]:F2}] -> {label} (output sigmoide: {raw:F6})");
}

Console.WriteLine();
Console.WriteLine("=== Fine ===");
