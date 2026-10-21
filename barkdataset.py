#!/usr/bin/env python3
"""
Build a tree-bark image dataset from iNaturalist research-grade observations.

Step 1 (see how many bark photos exist per species):
    python download_bark_dataset.py --count-only

Step 2 (download):
    python download_bark_dataset.py --out bark_data --max-per-species 400

Needs:  pip install requests
Only photos with CC0 / CC-BY / CC-BY-SA licences are kept. Attribution for every
image is written to metadata.csv - keep it and cite iNaturalist in your work.

NOTE: iNaturalist has no "bark" annotation. We search observations whose TAGS or
DESCRIPTION contain the word "bark" (and optionally "trunk"). Some downloaded photos
will still be leaves/flowers, so you MUST review/clean the images before training.
"""
import argparse
import csv
import re
import time
from pathlib import Path

import requests

API = "https://api.inaturalist.org/v1"
HEADERS = {"User-Agent": "bark-dataset-builder/1.0 (student research project)"}
ALLOWED_LICENSES = {"cc0", "cc-by", "cc-by-sa"}
KEYWORDS = ["bark", "trunk"]          # searched in tags and in descriptions
SEARCH_FIELDS = ["tags", "description"]
DELAY = 1.1                           # seconds between API calls (be polite)

# Corrected / current names. iNaturalist resolves synonyms, and the script prints
# the name it actually matched so you can confirm it.
# (scientific name, common name). The scientific name is tried first; if iNaturalist
# can't resolve it, the common name is used as a fallback.
SPECIES = [
    ("Peltophorum pterocarpum", "Copperpod"),            # = P. ferrugineum
    ("Mimusops elengi", "Spanish Cherry"),
    ("Azadirachta indica", "Neem"),
    ("Ficus benjamina", "Weeping Fig"),
    ("Bauhinia purpurea", "Purple Orchid Tree"),
    ("Millingtonia hortensis", "Indian Cork Tree"),      # the "Bignonia" in your list
    ("Syzygium cumini", "Jamun"),                        # = Eugenia jambolana
    ("Calophyllum inophyllum", "Alexandrian Laurel"),
    ("Tabebuia rosea", "Pink Trumpet Tree"),
    ("Lagerstroemia speciosa", "Queen's Crape Myrtle"),
    ("Alstonia scholaris", "Devil's Tree"),
]


def api_get(path, params):
    for attempt in range(5):
        try:
            r = requests.get(f"{API}{path}", params=params, headers=HEADERS, timeout=30)
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            time.sleep(DELAY)
            return r.json()
        except requests.RequestException as e:
            print(f"  request error ({e}); retrying...")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"API failed for {path}")


def resolve_taxon(name):
    data = api_get("/taxa/autocomplete", {"q": name, "rank": "species", "per_page": 5})
    results = data.get("results", [])
    if not results:
        return None
    t = results[0]
    return t["id"], t["name"]


def base_params(taxon_id, field, keyword):
    p = {
        "taxon_id": taxon_id,
        "quality_grade": "research",
        "photos": "true",
        "photo_license": "cc0,cc-by,cc-by-sa",
        "per_page": 200,
        "order_by": "id",
        "order": "desc",
    }
    if keyword:  # keyword=None means "all photos of this species"
        p["q"] = keyword
        p["search_on"] = field
    return p


def count_species(taxon_id):
    total = 0
    for field in SEARCH_FIELDS:
        for kw in KEYWORDS:
            p = base_params(taxon_id, field, kw)
            p["per_page"] = 1
            total += api_get("/observations", p).get("total_results", 0)
    return total  # upper bound: observations may match several queries


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def download_species(name, taxon_id, out_dir, max_photos, writer, seen, all_photos=False):
    folder = out_dir / slug(name)
    folder.mkdir(parents=True, exist_ok=True)
    saved = 0
    fields = [None] if all_photos else SEARCH_FIELDS
    keywords = [None] if all_photos else KEYWORDS
    for field in fields:
        for kw in keywords:
            page = 1
            while saved < max_photos:
                p = base_params(taxon_id, field, kw)
                p["page"] = page
                data = api_get("/observations", p)
                results = data.get("results", [])
                if not results:
                    break
                for obs in results:
                    for photo in obs.get("photos", []):
                        lic = (photo.get("license_code") or "").lower()
                        if lic not in ALLOWED_LICENSES or photo["id"] in seen:
                            continue
                        url = photo["url"].replace("square", "large")
                        fname = f"{obs['id']}_{photo['id']}.jpg"
                        try:
                            img = requests.get(url, headers=HEADERS, timeout=30)
                            img.raise_for_status()
                        except requests.RequestException:
                            continue
                        (folder / fname).write_bytes(img.content)
                        seen.add(photo["id"])
                        saved += 1
                        writer.writerow([name, fname, obs["id"], photo["id"], lic,
                                         photo.get("attribution", ""),
                                         f"https://www.inaturalist.org/observations/{obs['id']}"])
                        if saved >= max_photos:
                            break
                    if saved >= max_photos:
                        break
                page += 1
                if page * 200 > 10000:  # iNaturalist paging limit
                    break
        print(f"  {name}: {saved} photos so far")
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--out", default="bark_data")
    ap.add_argument("--max-per-species", type=int, default=400)
    ap.add_argument("--all-photos", action="store_true",
                    help="ignore the bark/trunk keywords and take every CC-licensed "
                         "research-grade photo; then run filter_bark_clip.py to keep only bark")
    args = ap.parse_args()

    resolved = []
    for name, common in SPECIES:
        r = resolve_taxon(name) or resolve_taxon(common)
        if r is None:
            print(f"!! Could not resolve '{name}' / '{common}' on iNaturalist - skipping")
            continue
        print(f"{name}  ->  matched: {r[1]} (taxon {r[0]})")
        resolved.append((r[1], r[0]))

    if args.count_only:
        print("\nObservations with 'bark'/'trunk' in tags or description (upper bound):")
        for name, tid in resolved:
            if args.all_photos:
                p = base_params(tid, None, None)
                p["per_page"] = 1
                print(f"  {name:30s} {api_get('/observations', p).get('total_results', 0)} observations (all photos)")
            else:
                print(f"  {name:30s} {count_species(tid)}")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    seen = set()
    with open(out_dir / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["species", "file", "observation_id", "photo_id", "license",
                    "attribution", "observation_url"])
        for name, tid in resolved:
            print(f"\nDownloading {name} ...")
            download_species(name, tid, out_dir, args.max_per_species, w, seen,
                             all_photos=args.all_photos)
    print("\nDone. Review images manually, remove non-bark photos, then split train/val/test "
          "by observation_id so photos of the same tree never land in two splits.")


if __name__ == "__main__":
    main()