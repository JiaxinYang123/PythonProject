# Sector-level DeFi exposure-contraction forecasting

## Overview

This project develops a reproducible Python pipeline for analysing and forecasting sector-level exposure contraction in decentralised finance (DeFi). It uses the weekly protocol-level exposure networks provided by the DeXposure dataset, aggregates them into seven analytical sectors, and evaluates one-week-ahead forecasts of the upper conditional quantile of sector-level exposure contraction.

The empirical analysis focuses on four forecasting sectors:

- Asset Management
- Infrastructure, Services & Financial Products
- Lending, Borrowing & Real World Assets
- Trading & Exchanges

Privacy & Security, Primary Market Tokens, and Other / Unknown remain in the sector network and contribute to the network predictors, but they are not used as forecast targets.

The principal forecasting exercise estimates the 90th conditional quantile. A 95th-quantile specification is included as a robustness analysis. The model results are interpreted as predictive evidence rather than causal estimates of contagion.

## Research questions

The code supports three research questions:

1. What sector-level patterns of exposure contraction, activity, and network conditions can be identified from the DeXposure graph time series?
2. To what extent can the next-week 90th conditional quantile of exposure contraction be forecast using observed historical information?
3. Do sector-level network conditions provide incremental predictive information beyond a sector's own contraction history and exposure activity?

## Project structure

```text
PythonProject/
|-- data/
|   |-- raw/                 # Original DeXposure network snapshots
|   |-- mapping/             # Official protocol and token mappings
|   `-- processed/           # Generated sector panels and predictors
|-- figures/                 # Generated dissertation figures
|-- outputs/
|   |-- tables/              # Descriptive and evaluation tables
|   |-- model_results/       # Model selection, predictions, and comparisons
|   `-- logs/                # Optional run logs
|-- references/              
|-- src/                     # main work
|-- README.md
`-- requirements.txt
```


## Data requirements

Mainly use the following benchmark files before running the pipeline:

```text
data/raw/historical-network_week_2020-03-30.json
data/mapping/id_to_info.json
data/mapping/token_to_protocol.json
```

The project may also retain `meta_df.csv`, `rev_map.json`, and the smaller network snapshot `historical-network_week_2025-07-01.json` for reference. The current main pipeline does not require these additional files.

The expected full sample contains 283 weekly snapshots from 23 March 2020 to 18 August 2025.

## Python environment

The project was developed with Python 3.13.

```text
python -m pip install -r requirements.txt
```

Since this project was carried out on the local PyCharm environment. A local `.venv` directory should be recreated on each computer. Therefore, of course, I did not upload the local virtual environment.

## Reproduction workflow

Run the numbered scripts in ascending order from the project root:

```text
python src/01_check_project.py
python src/02_check_data_files.py
python src/03_build_sector_panel.py
python src/04_validate.py
python src/05_descriptive_analysis.py
python src/06_build_forecasting_predictors.py
python src/07_knowledge_discovery_analysis.py
python src/08_select_model.py
python src/09_check.py
python src/10_fit_final_network_model.py
python src/11_analyse_final_network_forecasts.py
python src/12_compare_final_forecasting_models.py
python src/13_robustness_quantile_95.py
python src/14_create_sector_exposure_heatmap.py
```

The scripts can also be run individually in PyCharm. Each script resolves paths relative to the project root, so no machine-specific absolute path is required.

For a clean reproduction, remove the existing contents of `data/processed/`, `outputs/tables/`, `outputs/model_results/`, `outputs/logs/`, and `figures/` before running the pipeline. Do not delete anything from `data/raw/` or `data/mapping/`.

## Script guide

| Script | Purpose | Main outputs |
|---|---|---|
| `01_check_project.py` | Checks that the required project directories  | Console check |
| `02_check_data_files.py` | Checks the main raw network and official mapping files | Console check |
| `03_build_sector_panel.py` | Streams the raw DeXposure data, maps endpoints to seven sectors, and constructs the weekly sector matrix and panel | `sector_exposure_matrix.csv`, `sector_week_panel.csv`, validation and mapping summaries |
| `04_validate.py` | Verifies dimensions, flow reconstruction, contraction scores, and exposure conservation | Console validation |
| `05_descriptive_analysis.py` | Summarises contraction-score distributions for the four forecast sectors | Table 2 data and Figure 1 |
| `06_build_forecasting_predictors.py` | Constructs lagged contraction, activity, network-pressure, and concentration predictors | `forecasting_panel.csv` |
| `07_knowledge_discovery_analysis.py` | Summarises sector activity and compares normal and upper-tail states | Tables 3 and 4 |
| `08_select_model.py` | Selects the historical window and quantile-regression lag orders using the validation sample | Table 5 and selected specifications |
| `09_check.py` | Checks the selected specifications and final forecasting sample | Console check |
| `10_fit_final_network_model.py` | Re-estimates the selected network model and produces test-period forecasts | `final_network_test_predictions.csv` |
| `11_analyse_final_network_forecasts.py` | Evaluates the final network forecasts overall and by sector | Tables 6 and 7, and Figure 3 |
| `12_compare_final_forecasting_models.py` | Compares the historical, time-series, and network forecasting models | Tables 8 and 9, and Figures 4 and 5 |
| `13_robustness_quantile_95.py` | Repeats model selection and test evaluation at the 95th conditional quantile | Table 10 and supporting predictions |
| `14_create_sector_exposure_heatmap.py` | Calculates full-sample directed sector-exposure shares | Figure 2 and supporting matrices |

The helper module `dexposure_io.py` streams the large JSON network file. The helper module `dexposure_mapping.py` applies the official DeXposure mappings and the documented seven-sector crosswalk.

## Paper-output mapping

| Dissertation item | Script | Output file |
|---|---|---|
| Table 1 | `03_build_sector_panel.py` | `outputs/tables/sector_network_validation.csv` |
| Table 2 | `05_descriptive_analysis.py` | `outputs/tables/contraction_descriptive_statistics.csv` |
| Figure 1 | `05_descriptive_analysis.py` | `figures/contraction_distribution.png` |
| Figure 2 | `14_create_sector_exposure_heatmap.py` | `figures/sector_exposure_heatmap.png` |
| Table 3 | `07_knowledge_discovery_analysis.py` | `outputs/tables/sector_activity_network_summary.csv` |
| Table 4 | `07_knowledge_discovery_analysis.py` | `outputs/tables/sector_tail_state_comparison.csv` |
| Table 5 | `08_select_model.py` | `outputs/model_results/validation_model_selection.csv` |
| Table 6 | `11_analyse_final_network_forecasts.py` | `outputs/tables/network_test_overall_performance.csv` |
| Figure 3 | `11_analyse_final_network_forecasts.py` | `figures/final_network_forecasts_over_time.png` |
| Table 7 | `11_analyse_final_network_forecasts.py` | `outputs/tables/network_test_sector_performance.csv` |
| Table 8 | `12_compare_final_forecasting_models.py` | `outputs/model_results/final_model_overall_comparison.csv` |
| Table 9 | `12_compare_final_forecasting_models.py` | `outputs/model_results/sector_network_incremental_value.csv` |
| Figure 4 | `12_compare_final_forecasting_models.py` | `figures/sector_forecasting_model_comparison.png` |
| Figure 5 | `12_compare_final_forecasting_models.py` | `figures/cumulative_network_forecasting_gain.png` |
| Table 10 | `13_robustness_quantile_95.py` | `outputs/model_results/q95_robustness_results.csv` |

Additional CSV files contain intermediate predictions, validation results, mapping summaries, or weekly loss differences used to produce the reported results.

## Expected validation results

A successful full run should reproduce the following core dimensions:

- 283 weekly network snapshots
- 7 analytical sectors
- 13,867 sector-pair observations
- 1,981 sector-week observations
- 282 valid contraction-score observations for each forecast sector
- 980 complete training-validation observations for the selected four-lag models
- 132 test observations across 33 forecast weeks and four sectors
- Selected specifications: `m = 52`, `p_TS = 4`, and `p_Net = 4`



## Notes

- The data split is chronological rather than random.
- Test-period outcomes are not used for model selection or coefficient estimation.
- Network variables are constructed using all seven analytical sectors.
- It should be run from 01 to 14 in sequence.

