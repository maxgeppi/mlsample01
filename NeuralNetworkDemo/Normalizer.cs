namespace NeuralNetworkDemo;

/// <summary>
/// Converte i valori reali inseriti dall'utente nei range 0-1
/// necessari per la rete neurale.
/// </summary>
public static class Normalizer
{
    // Range reali per il peso dei frutti (grammi)
    public const double PesoMinimo = 50.0;   // mela molto piccola
    public const double PesoMassimo = 300.0;  // arancia molto grande

    // Mappa colore testuale -> valore normalizzato
    private static readonly Dictionary<string, double> ColoriMap = new(StringComparer.OrdinalIgnoreCase)
    {
        { "verde",          0.0 },
        { "verde chiaro",   0.15 },
        { "giallo verde",   0.25 },
        { "giallo",         0.40 },
        { "giallo arancio", 0.60 },
        { "arancio chiaro", 0.75 },
        { "arancione",      0.90 },
        { "arancio scuro",  1.0 }
    };

    /// <summary>
    /// Normalizza il peso in grammi nel range 0-1.
    /// </summary>
    public static double NormalizzaPeso(double pesoGrammi)
    {
        double normalizzato = (pesoGrammi - PesoMinimo) / (PesoMassimo - PesoMinimo);
        return Math.Clamp(normalizzato, 0.0, 1.0);
    }

    /// <summary>
    /// Converte il nome del colore in un valore 0-1.
    /// Restituisce null se il colore non e' riconosciuto.
    /// </summary>
    public static double? NormalizzaColore(string colore)
    {
        if (ColoriMap.TryGetValue(colore.Trim(), out double valore))
            return valore;
        return null;
    }

    /// <summary>
    /// Restituisce i colori disponibili con i rispettivi valori.
    /// </summary>
    public static IReadOnlyDictionary<string, double> ColoriDisponibili => ColoriMap;
}
