# Data

## `raw/bengaluru_house_prices.csv` (training data, committed)

| | |
|---|---|
| Dataset | "Bengaluru House price data" by amitabhajoy on Kaggle |
| Kaggle page | https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data |
| Licence | CC0: Public Domain, as declared on Kaggle (checked via the Kaggle dataset API, October 2026) |
| Copy used here | Public mirror `https://raw.githubusercontent.com/codebasics/py/master/DataScience/BangloreHomePrices/model/bengaluru_house_prices.csv` |
| SHA-256 | `23ddf71be7d79a259b62e125b9d56cb531f9d051b0da650c0fe7cc82365a1015` |
| Rows | 13,320 listings, 9 columns |
| Columns | `area_type, availability, location, size, society, total_sqft, bath, balcony, price` (`price` is in lakh rupees) |

Provenance caveats:

* The dataset is widely reported to originate from a MachineHack (Analytics India) Bengaluru house-price hackathon. The Kaggle uploader declares CC0; the original collector's terms are not published, so if you plan anything beyond coursework, check them yourself.
* The collection date is not documented. Possession dates such as "18-Dec" suggest listings from about 2017-18. Prices are past **asking** prices from listings, not registered sale prices, and they are not today's market prices.
* The `society` column is not used.

Re-download and verify with `python scripts/download_dataset.py`.

## `live/` (not committed)

Created by `python data_fetcher.py`. See `live/README.md`.
