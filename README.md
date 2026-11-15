# Demand Forecasting & Inventory Replenishment

VISUELLE 2.0 proof of concept for demand forecasting and replenishment. The
prototype uses external data; it does not use SHEIN internal data.

## Requirements

- Python 3.11 or newer

Install dependencies and launch the Streamlit app:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

## Input data

By default, the pipeline expects `sales.csv`, `restocks.csv`,
`price_discount_series.csv`, `vis2_gtrends_data.csv`, and
`vis2_weather_data.csv` in `data/raw/`. Their names and the data directory are
configured in `config/settings.yaml`. The source CSVs are not included in this
repository; only publish data you have permission to share.

`python -m codes.sample_data` can generate deterministic synthetic data for
self-tests. Such data and any figures derived from them are not results and
must not be used for reporting or operational decisions.

## Tests

```powershell
python -m pytest
```

## Project notes

- [Run and replaceability guide](docs/RUN_AND_REPLACEABILITY.md)
- [Decision packet](docs/DECISION_PACKET.md)
