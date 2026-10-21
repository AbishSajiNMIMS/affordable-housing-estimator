Live listings fetched by `python data_fetcher.py` are saved here (`live_listings.csv`, `fetch_status.json`, `raw_cache/`).

They are not committed: the listings come from 99acres via the Scrappa API, and their terms do not grant a licence to redistribute the data. On AWS this folder is mounted on Amazon EFS so it survives redeployments.
