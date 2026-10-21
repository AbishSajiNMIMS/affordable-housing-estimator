"""Check why listing photos or area maps do not appear on the result cards.

Run:  python check_media.py
Prints what it finds; never prints your keys.
"""
import pandas as pd
import requests

import config


def check_maps():
    print("== Area maps ==")
    if not config.GOOGLE_MAPS_API_KEY:
        print("GOOGLE_MAPS_API_KEY is empty. Put it in .env (next to app.py), save, then restart the app.")
        return
    print(f"GOOGLE_MAPS_API_KEY is set ({len(config.GOOGLE_MAPS_API_KEY)} characters).")
    try:
        resp = requests.get("https://maps.googleapis.com/maps/api/staticmap",
                            params={"center": "Andheri West, Mumbai, India", "zoom": 14, "size": "100x100",
                                    "key": config.GOOGLE_MAPS_API_KEY}, timeout=15)
    except requests.RequestException as exc:
        print(f"Could not reach Google: {exc.__class__.__name__}")
        return
    if resp.status_code == 200 and resp.headers.get("Content-Type", "").startswith("image/"):
        print("Google returned a map image: the key works. Restart the app if maps still do not show.")
    else:
        print(f"Google refused (HTTP {resp.status_code}): {' '.join(resp.text.split())[:300]}")
        print("Usual fixes: enable 'Maps Static API' for this key's project, make sure billing is linked to "
              "the project, and if the key has API restrictions include Maps Static API. "
              "Changes can take a few minutes to apply.")


def check_photos():
    print("\n== Mumbai listing photos ==")
    path = config.CITIES["mumbai"]["data_path"]
    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        print(f"No Mumbai data at {path}.")
        return
    if "photo_url" not in data.columns:
        print("Your Mumbai CSV has no photo column. Run: python apify_fetcher.py --from-cache "
              "then: python train_model.py --city mumbai")
        return
    photos = data["photo_url"].fillna("").astype(str)
    with_photo = photos[photos.str.startswith("https://")]
    print(f"{len(with_photo)} of {len(data)} saved listings have a photo link.")
    ref = config.city_artifacts_dir("mumbai") / "reference_data.csv"
    if ref.exists():
        ref_df = pd.read_csv(ref)
        n = int(ref_df.get("photo_url", pd.Series(dtype=str)).fillna("").astype(str).str.startswith("https://").sum())
        print(f"The trained Mumbai model has {n} listings with photos.")
        if n == 0 and len(with_photo):
            print("Retrain so the website sees them: python train_model.py --city mumbai")
    if len(with_photo) == 0:
        print("The Apify data had no photo links (checked fields: photo_url, image, image_url, thumbnail, cover_image, photos, images).")
        return
    url = with_photo.iloc[0]
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        kind = resp.headers.get("Content-Type", "")
        print(f"Example photo: HTTP {resp.status_code}, {kind or 'no content type'}  {url[:90]}")
        if resp.status_code != 200 or not kind.startswith("image/"):
            print("Housing.com did not return an image for this link, so browsers will not show it either.")
    except requests.RequestException as exc:
        print(f"Could not load the example photo: {exc.__class__.__name__}")


if __name__ == "__main__":
    check_maps()
    check_photos()
