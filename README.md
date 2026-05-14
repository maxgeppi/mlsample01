# Neural Network Demo in C#

Esempio semplice di rete neurale in C# con classi, che utilizza la funzione **sigmoide** per classificare due tipi di oggetti basandosi su due caratteristiche.

## Scenario

Classificazione di frutti: **MELA** vs **ARANCIA**

| Caratteristica | Descrizione |
|---|---|
| Peso normalizzato | Valore tra 0 e 1 |
| Colore | Scala 0-1 (0 = verde, 1 = arancione) |

## Architettura della Rete

```
Input (2) -> Strato Nascosto (3 neuroni, sigmoide) -> Output (1 neurone, sigmoide)
```

## Classi

- **`Neuron`**: Rappresenta un singolo neurone con pesi, bias e attivazione sigmoide
- **`NeuralNetwork`**: Rete neurale con uno strato nascosto, addestramento con backpropagation
- **`Program`**: Punto di ingresso con dati di esempio, addestramento e predizioni

## Come eseguire

```bash
cd NeuralNetworkDemo
dotnet run
```

## Requisiti

- .NET 8.0 SDK
