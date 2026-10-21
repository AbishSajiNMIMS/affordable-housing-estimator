# Mumbai listings (not in Git)

`apify_fetcher.py` saves current Mumbai sale listings from Housing.com here,
fetched with **your own** Apify token through the
[Housing.com Scraper actor](https://apify.com/themineworks/housing-com-scraper):

| File | What it is |
|---|---|
| `housing_com_mumbai.csv` | validated listings in the project schema (training data for the Mumbai model) |
| `fetch_status.json` | time of the last fetch attempt / success and the last error |
| `raw_cache/apify_mumbai_latest.json` | the raw response from the last fetch, for checking the parser |

These files are git-ignored: the listings are Housing.com content and are
not redistributable. Each row keeps its listing URL and the time it was
fetched, and prices are **asking** prices, not sale prices.

```bash
python apify_fetcher.py --max 1000        # needs APIFY_TOKEN in .env
python train_model.py --city mumbai       # needs at least 300 valid listings
```
