# Sample Dataset

This folder contains a small subset of the full stock dataset — enough to run the web app and explore all features.

## Contents

| Exchange | Tickers included |
|---|---|
| **NSE** | TCS, INFY, RELIANCE, HDFCBANK |
| **BSE** | 08ABB, 08ADD, 08ADR, 08AGG, 08AMD |

Each ticker contains all available yearly CSVs (2000–2023) with actual data.

## How to use

The app looks for data in `comp_stock_data/` by default.  
To run with sample data, create a symlink or copy this folder:

```bash
# Windows PowerShell
Copy-Item sample_data comp_stock_data -Recurse

# Or on Linux/macOS
cp -r sample_data comp_stock_data
```

Then start the server normally:

```bash
python manage.py runserver
```

> **Note:** With only 5 NSE tickers, ranking queries like `top 10 NSE` will return at most 5 results. That's expected with sample data.

## Full dataset

The complete dataset (~150,000 CSV files covering 1,800+ NSE and 4,400+ BSE stocks from 2000–2023) is not included in this repo due to its size.
