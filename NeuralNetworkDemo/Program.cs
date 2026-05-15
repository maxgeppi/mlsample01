using NeuralNetworkDemo;

// ============================================================
// Esempio di rete neurale per classificare due tipi di oggetti
// basandosi su due caratteristiche (proprieta).
//
// Scenario: classificare frutti in MELE (0) e ARANCE (1)
//   Caratteristica 1: Peso in grammi (normalizzato automaticamente)
//   Caratteristica 2: Colore (testo -> valore normalizzato)
//
// Architettura: 2 input -> 3 neuroni nascosti -> 1 output
// Attivazione: Sigmoide
// ============================================================

Console.WriteLine("=== Rete Neurale con Sigmoide in C# ===");
Console.WriteLine();
Console.WriteLine("Classificazione di frutti: MELA vs ARANCIA");
Console.WriteLine($"  Peso: da {Normalizer.PesoMinimo}g a {Normalizer.PesoMassimo}g (normalizzato in 0-1)");
Console.WriteLine("  Colore: testo libero convertito in scala 0-1");
Console.WriteLine();

// Dati di addestramento con valori gia normalizzati
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

// Creazione e addestramento della rete neurale
var network = new NeuralNetwork(
    numberOfInputs: 2,
    hiddenNeurons: 3,
    learningRate: 0.5,
    seed: 42
);

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

// --- Modalita interattiva ---
Console.WriteLine();
Console.WriteLine("=============================================");
Console.WriteLine("  MODALITA' INTERATTIVA");
Console.WriteLine("  Inserisci i dati reali del frutto e la");
Console.WriteLine("  rete neurale lo classifichera per te!");
Console.WriteLine("=============================================");
Console.WriteLine();
Console.WriteLine("Colori disponibili:");
foreach (var colore in Normalizer.ColoriDisponibili)
{
    Console.WriteLine($"  - {colore.Key,-20} (valore: {colore.Value:F2})");
}
Console.WriteLine();

while (true)
{
    Console.WriteLine("--- Nuova classificazione (scrivi 'esci' per uscire) ---");
    Console.WriteLine();

    // Input peso
    Console.Write($"Inserisci il peso in grammi ({Normalizer.PesoMinimo}-{Normalizer.PesoMassimo}): ");
    string? inputPeso = Console.ReadLine();
    if (string.IsNullOrWhiteSpace(inputPeso) || inputPeso.Trim().Equals("esci", StringComparison.OrdinalIgnoreCase))
        break;

    if (!double.TryParse(inputPeso, out double pesoGrammi))
    {
        Console.WriteLine("  Errore: inserisci un numero valido per il peso.");
        Console.WriteLine();
        continue;
    }

    // Input colore
    Console.Write("Inserisci il colore (es. verde, giallo, arancione): ");
    string? inputColore = Console.ReadLine();
    if (string.IsNullOrWhiteSpace(inputColore) || inputColore.Trim().Equals("esci", StringComparison.OrdinalIgnoreCase))
        break;

    double? coloreNorm = Normalizer.NormalizzaColore(inputColore);
    if (coloreNorm == null)
    {
        Console.WriteLine($"  Errore: colore '{inputColore.Trim()}' non riconosciuto.");
        Console.WriteLine("  Usa uno dei colori elencati sopra.");
        Console.WriteLine();
        continue;
    }

    // Normalizzazione e predizione
    double pesoNorm = Normalizer.NormalizzaPeso(pesoGrammi);

    Console.WriteLine();
    Console.WriteLine($"  Valori inseriti:     peso = {pesoGrammi}g, colore = {inputColore.Trim()}");
    Console.WriteLine($"  Valori normalizzati: peso = {pesoNorm:F4}, colore = {coloreNorm.Value:F4}");

    double[] input = [pesoNorm, coloreNorm.Value];
    double rawOutput = network.Forward(input);
    int prediction = network.Predict(input);
    string risultato = prediction == 0 ? "MELA" : "ARANCIA";

    Console.WriteLine();
    Console.WriteLine($"  >>> RISULTATO: {risultato} (sigmoide: {rawOutput:F6}, confidenza: {(prediction == 1 ? rawOutput : 1 - rawOutput) * 100:F1}%)");
    Console.WriteLine();
}

Console.WriteLine();
Console.WriteLine("=== Fine ===");
