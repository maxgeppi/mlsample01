---
name: testing-neural-network
description: Test the C# neural network demo end-to-end. Use when verifying changes to Neuron, NeuralNetwork, Normalizer, or Program.cs.
---

# Testing the Neural Network Demo

## Prerequisites

- .NET 8.0 SDK installed (`dotnet-sdk-8.0`)
- Blueprint handles installation automatically

## Build

```bash
cd NeuralNetworkDemo && dotnet build
```

Build must succeed with 0 errors and 0 warnings.

## Running

This is a **console app** — all testing is done via shell commands with piped input. No browser/GUI recording needed.

```bash
cd NeuralNetworkDemo && dotnet run
```

The app has two phases:
1. **Training phase**: Automatic, prints epoch errors every 1000 epochs
2. **Interactive mode**: User enters weight (grams) and color (text), app normalizes and classifies

## Testing with Piped Input

Pipe inputs line by line (weight, then color, repeat, end with "esci"):

```bash
echo -e "80\nverde\n250\narancione\nesci" | dotnet run
```

## Key Test Cases

### Normalization Math
Formula: `(peso - 50) / (300 - 50)`, clamped to [0, 1]
- 50g → 0.0000
- 300g → 1.0000
- 175g → 0.5000
- Below-min (e.g. 10g) → clamped to 0.0000
- Above-max (e.g. 500g) → clamped to 1.0000

### Color Mapping
8 supported colors (case-insensitive): verde (0.0), verde chiaro (0.15), giallo verde (0.25), giallo (0.40), giallo arancio (0.60), arancio chiaro (0.75), arancione (0.90), arancio scuro (1.0)

### Classification
- Low weight + green → MELA (sigmoide < 0.5)
- High weight + orange → ARANCIA (sigmoide > 0.5)

### Error Handling
- Non-numeric weight (e.g. "abc") → error message, loop continues
- Unknown color (e.g. "blu") → error message, loop continues
- "esci" → clean exit with `=== Fine ===`

### Training Phase
- Epoch 5000 error should be < 0.001
- All 10 training samples must classify correctly (5 MELA, 5 ARANCIA)
- Output is deterministic (seed 42)

## Notes

- No CI is configured on this repo
- The seed is fixed at 42 for reproducibility — all runs produce identical training output
- Colors are case-insensitive (StringComparer.OrdinalIgnoreCase)
