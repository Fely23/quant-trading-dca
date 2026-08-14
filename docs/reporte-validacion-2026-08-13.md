# Reporte de validación — quant-trading-dca

**Fecha de ejecución:** 13 de agosto de 2026

**Fuente:** ejecución local de `run_experiment.py` y `dynamic_rebalancing.py`

**Horizonte:** clasificador con ~3 años de datos; simulador DCA con 24 aportes mensuales de USD 500
**Estado:** análisis de investigación; no apto para ejecutar operaciones reales.

## Conclusión ejecutiva

El clasificador ahora se evalúa con una secuencia *walk-forward* estricta: cada fold entrena únicamente con información anterior a su periodo de prueba. Bajo esa evaluación más realista, no demuestra ventaja económica: su rentabilidad neta acumulada fue **-2.16%**, frente a **49.68%** de comprar y mantener el S&P 500.

En la simulación DCA, Momentum Dinámico obtuvo el mayor retorno en esta ventana, pero rota mensualmente entre tres activos y tiene exposición recurrente a BTC. Mínima Varianza permaneció prácticamente por completo en `SHV`, por lo que redujo volatilidad a costa de retorno.

**Decisión recomendada:** no usar el clasificador como señal operativa. Mantener el simulador como herramienta de investigación y validar Momentum en varios periodos independientes antes de asignar capital.

## Indicadores clave

### Clasificador del S&P 500

| Indicador | Resultado | Lectura |
| --- | ---: | --- |
| ROC-AUC | 0.6204 | Señal predictiva moderada, insuficiente por sí sola. |
| Accuracy | 69.26% | Parcialmente impulsada por la mayor frecuencia de clase negativa. |
| Recall de señal positiva | 17.65% | Detecta pocas oportunidades positivas. |
| Días evaluados | 605 | Muestra limitada a ~3 años. |
| Operaciones | 24 | Actividad baja, con costes todavía relevantes. |
| Retorno acumulado neto | **-2.16%** | Negativo. |
| Sharpe neto anualizado | -0.0385 | No compensa el riesgo. |
| Máx. drawdown neto | -15.74% | Reducción material de capital. |
| Buy & Hold: retorno acumulado | **49.68%** | Benchmark claramente superior. |
| Buy & Hold: Sharpe | 1.1433 | Mejor relación retorno/riesgo en este periodo. |

### Simulación DCA

| Estrategia | Aportado | Valor final | ROI | TIR anual | Máx. DD | Sharpe | Lectura |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Moderado Estático | USD 12,000 | USD 15,378.84 | 28.16% | 21.64% | -14.08% | 1.379 | Referencia simple y diversificada. |
| Momentum Dinámico | USD 12,000 | USD 17,231.02 | **43.59%** | **32.64%** | -14.78% | 1.344 | Mejor retorno en la muestra; requiere validación robusta. |
| Mínima Varianza | USD 12,000 | USD 12,596.82 | 4.97% | 3.99% | -0.03% | 18.771 | Casi toda la cartera quedó en `SHV`. |

## Validación aplicada

El esquema usa cinco folds temporales. El tamaño de entrenamiento aumenta progresivamente: **117 → 238 → 359 → 480 → 601** observaciones. Las observaciones cuyas etiquetas de retorno futuro se solapan con el inicio del test se excluyen del entrenamiento.

El cambio elimina el uso de datos futuros en los folds. El AUC cayó de 0.6463 a 0.6204, coherente con una evaluación menos optimista.

## Diagnóstico

El clasificador tuvo 30 verdaderos positivos y 140 falsos negativos. La señal no se tradujo en una estrategia rentable y quedó muy por debajo del benchmark.

Momentum asigna aproximadamente 33.33% a cada uno de los tres activos con mejor retorno en los 60 días previos. Su resultado es atractivo, pero la muestra de 24 meses es corta y su riesgo depende de selección frecuente, incluidos activos volátiles.

El optimizador de mínima varianza no tenía límites de concentración; por ello eligió `SHV` con aproximadamente 99.5%–100% de la cartera en casi todos los meses. Es una posición de bajo riesgo, no una cartera diversificada de mínima varianza.

## Riesgos y limitaciones

- Resultados históricos e hipotéticos; no incluyen impuestos ni todos los costes reales de ejecución.
- `yfinance` y los precios disponibles al descargar pueden producir resultados diferentes en otras ejecuciones.
- La ventana DCA de 24 meses no cubre suficientes regímenes de mercado.
- Los parámetros (60 días de *lookback*, tres activos, 5 bps de coste) son supuestos de investigación.

## Próximas acciones

| Prioridad | Acción | Resultado esperado |
| --- | --- | --- |
| Alta | No usar el clasificador como señal operativa. | Evitar una estrategia con retorno neto y Sharpe negativos. |
| Alta | Probar DCA en 2010–2015, 2015–2020, 2020–2022 y 2022–2026. | Evaluar estabilidad en distintos regímenes de mercado. |
| Alta | Añadir un máximo de concentración, por ejemplo 40% por activo, a mínima varianza. | Evitar soluciones casi 100% `SHV`. |
| Media | Probar Momentum con costes de 10–25 bps y sin BTC. | Medir sensibilidad a ejecución y concentración. |
| Media | Guardar fechas, parámetros, versión de código y resultados de cada corrida. | Asegurar trazabilidad. |

## Parámetros de la corrida

```text
Clasificador: years=3, horizon=5, n_splits=5, percentile=0.70, tx_cost=0.0005
Simulador DCA: aporte mensual=USD 500, meses=24, lookback=60, tx_cost=0.0005
Universo: QQQ, SMH, BTC-USD, SPY, GLD, IEF, USMV, XLV, SHV
```

> Este reporte analiza código y simulaciones. No es asesoría de inversión ni una recomendación de compra, venta o asignación de activos.
