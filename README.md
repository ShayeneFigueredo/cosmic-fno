# CosmicFNO: Emulating Gravitational Waves with Fourier Neural Operators

**CosmicFNO** is a Scientific Initiation project applying **Fourier Neural Operators (FNO)** to model and emulate gravitational waves from binary black hole mergers, specifically focusing on the dominant $h_{22}$ mode.

## Motivation
Traditional neural networks (like Convolutional Neural Networks - CNNs) struggle with high-frequency, chaotic signals such as the merger and ringdown phases of a gravitational wave. By operating in the frequency domain, FNOs can capture global continuous patterns rather than being constrained by local grid resolutions.

## Baseline Comparison: FNO vs 1D-CNN (Sanity Check)
To validate the architecture, we performed a Representation Learning test (overfitting on a single real astrophysical signal from the SXS catalog: `SXS:BBH:0305` - GW150914). Both models were trained for 100 epochs under identical conditions.

### 1D-CNN (Baseline)
- **MSE Error:** ~0.0164
- **Analysis:** The CNN acts "myopic" due to its local receptive field. It smooths out the signal and completely misses the high-frequency amplitude peaks during the collision.
![CNN Result](docs/assets/resultado_treino_cnn.png)

### Fourier Neural Operator (FNO)
- **MSE Error:** ~0.0003
- **Analysis:** By using the Fast Fourier Transform (FFT), the FNO achieves a global receptive field. It successfully captures the complex wave physics, mapping the exact frequency and amplitude of the collision.
![FNO Result](docs/assets/resultado_treino_real.png)

## Next Steps
- Generate a larger, diverse dataset of SXS simulations.
- Train the FNO to generalize across different mass ratios and spins.
- Evaluate the zero-shot super-resolution capabilities of the FNO (inferring at higher temporal resolutions than the training data).

