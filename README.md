# S&P 500 ML Classifier & Dynamic Portfolio DCA Simulator

Este repositorio contiene dos pipelines cuantitativos desarrollados en Python:
1. **Clasificador S&P 500 (`run_experiment.py`)**: Clasificación de retornos futuros a 5 días libre de *look-ahead bias* utilizando un esquema de validación cruzada purgada y con embargo (`PurgedTimeSeriesSplit`).
2. **Simulador de Portafolios DCA (`dynamic_rebalancing.py`)**: Simulador de inversión mensual recurrente (DCA) que compara el rebalanceo estático frente a algoritmos de rebalanceo dinámico basados en **Momentum Relativo** y **Mínima Varianza (SciPy Optimization)**.

## Estructura del Proyecto
* `run_experiment.py`: Script principal para ejecutar el clasificador de machine learning sobre el S&P 500.
* `dynamic_rebalancing.py`: Script principal para simular la inversión DCA y el rebalanceo de portafolios.
* `src/`:
  * `data.py`: Módulo de descarga y preparación de retornos futuros con `yfinance`.
  * `features.py`: Cálculo vectorizado de indicadores técnicos clásicos (RSI, MACD, Bollinger Bands, Volatilidad, etc.).
  * `validation.py`: Implementación matemática de la validación cruzada purgada y con embargo.
  * `pipeline.py`: Flujo de entrenamiento, escalamiento local intra-fold y backtesting del modelo ML.
* `requirements.txt`: Dependencias del proyecto.
* `.gitignore`: Exclusión de entornos virtuales y archivos locales temporales.

## Ejecución del Proyecto

1. **Clonar e ingresar a la carpeta**:
   ```bash
   cd quant_ml_trading
   ```

2. **Crear e instalar dependencias**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Ejecutar simulaciones**:
   * Clasificador: `python run_experiment.py`
   * Portafolios: `python dynamic_rebalancing.py`
