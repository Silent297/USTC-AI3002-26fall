# -*- coding: utf-8 -*-
"""C3：GeoNames 数据扩增数据集构建。

数据源（官方 dump，本脚本只读取本地文件）：
    https://download.geonames.org/export/dump/cities15000.zip
    https://download.geonames.org/export/dump/countryInfo.txt

协议（对齐指南2 §1 的规模，细节在 meta.json 中如实记录）：
    - 从 cities15000 的 populated places 中按 seed=20261004 抽样 12000 条；
    - 预留 30 个中等规模国家作为 test_unseen（1518 条），其国家码不出现在
      train/valid/test；
    - 其余国家抽 10482 条，随机切为 train 8382 / valid 1050 / test 1050。

Prompt 与课程 world 数据同模板：
    "What's the longitude and latitude of {place}, {country}?"
    （地名与国家名相同，如 Singapore / Vatican City，则省略国家后缀，
      与 datasets/world 的口径一致）

额外产出：
    - 原始 world train/valid 的 prompt 副本（供训练脚本统一读取）；
    - 原始 world valid 与 GeoNames 城市的 name+ISO 精确匹配标志
      （match_orig_valid），用于训练“无泄漏”增广集 new_noleak；
    - leak_audit.json / meta.json。

用法：
    python -X utf8 experiments/C3/c3_build_dataset.py
"""
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).with_name("data")
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "splits"
SEED = 20261004
TOTAL = 12000
TRAIN_N, VALID_N, TEST_N, UNSEEN_N = 8382, 1050, 1050, 1518
UNSEEN_COUNTRIES = 30
UNSEEN_MIN_CITIES, UNSEEN_MAX_CITIES = 8, 500
PROMPT = "What's the longitude and latitude of {place}?"
PROMPT_WITH_COUNTRY = "What's the longitude and latitude of {place}, {country}?"

# GeoNames countryInfo 的官方名与 datasets/world 常用名不一致时统一成后者
COUNTRY_ALIASES = {
    "Korea, Republic of": "South Korea",
    "Korea, Democratic People's Republic of": "North Korea",
    "Türkiye": "Turkey",
    "Turkey": "Turkey",
    "Viet Nam": "Vietnam",
    "Bolivia, Plurinational State of": "Bolivia",
    "Venezuela, Bolivarian Republic of": "Venezuela",
    "Tanzania, United Republic of": "Tanzania",
    "Moldova, Republic of": "Moldova",
    "Syrian Arab Republic": "Syria",
    "Lao People's Democratic Republic": "Laos",
    "Iran, Islamic Republic of": "Iran",
    "Russian Federation": "Russia",
    "Congo, The Democratic Republic of the": "Democratic Republic of the Congo",
    "Congo": "Republic of the Congo",
    "Côte d'Ivoire": "Côte d’Ivoire",
    "Palestine, State of": "Palestine",
    "Micronesia, Federated States of": "Micronesia",
    "Brunei Darussalam": "Brunei",
    "Cabo Verde": "Cape Verde",
    "Swaziland": "Eswatini",
    "Macedonia": "North Macedonia",
    "Burma": "Myanmar",
    "Bahamas": "The Bahamas",
    "Gambia": "The Gambia",
    "Czech Republic": "Czechia",
    "Ivory Coast": "Côte d’Ivoire",
    "Palestinian Territory": "Palestine",
    "The Netherlands": "Netherlands",
    "Timor Leste": "Timor-Leste",
}
# 原始 world 的复合国家标签需要拆开映射（只影响泄漏审计）
COUNTRY_EXTRA_ALIASES = {
    "DRC": "Democratic Republic of the Congo",
    "Côte d’Ivoire": "Côte d’Ivoire",
    "North Korea": "Korea, Democratic People's Republic of",
    "South Korea": "Korea, Republic of",
}


def normalize_name(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower().replace("’", "'").replace("–", "-")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[4:] if text.startswith("the ") else text


def parse_country_info(path: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5:
                continue
            names[fields[0]] = fields[4]
    if not names:
        raise ValueError(f"no countries parsed from {path}")
    return names


def display_country(iso: str, official: dict[str, str]) -> str:
    name = official.get(iso, iso).strip()
    return COUNTRY_ALIASES.get(name, name).strip()


def read_cities(path: Path, official: dict[str, str]) -> list[dict]:
    cities = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 19 or fields[6] != "P" or not fields[8]:
                continue
            try:
                lat, lon = float(fields[4]), float(fields[5])
            except ValueError:
                continue
            name = (fields[2] or fields[1]).strip()  # asciiname, fallback name
            if not name:
                continue
            cities.append(
                {
                    "geonameid": int(fields[0]),
                    "place": name,
                    "country": display_country(fields[8], official),
                    "iso": fields[8],
                    "latitude": lat,
                    "longitude": lon,
                    "population": int(fields[14] or 0),
                }
            )
    return cities


def world_place(sentence: str, country: str) -> str:
    prefix = "What's the longitude and latitude of "
    if not sentence.startswith(prefix) or not sentence.endswith("?"):
        raise ValueError(f"unexpected world sentence: {sentence!r}")
    body = sentence[len(prefix) : -1]
    suffix = f", {country}"
    if body.endswith(suffix):
        body = body[: -len(suffix)]
    return body.strip()


def build_country_iso(official: dict[str, str]) -> dict[str, str]:
    lookup = {}
    for iso, name in official.items():
        lookup[normalize_name(name)] = iso
        lookup[normalize_name(display_country(iso, official))] = iso
    for alias, target in COUNTRY_EXTRA_ALIASES.items():
        iso = next((code for code, name in official.items() if name == target), None)
        if iso:
            lookup[normalize_name(alias)] = iso
    return lookup


def world_rows(path: Path, country_iso: dict[str, str]) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            sentence, country = row["sentence"], row["country"]
            place = world_place(sentence, country)
            isos = set()
            for part in country.split("/"):
                iso = country_iso.get(normalize_name(part.strip()))
                if iso:
                    isos.add(iso)
            rows.append(
                {
                    "sentence": sentence,
                    "country": country,
                    "place": place,
                    "iso": sorted(isos),
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                }
            )
    return rows


def name_key(place: str) -> str:
    return normalize_name(place)


def prompt_for(city: dict) -> str:
    if name_key(city["place"]) == name_key(city["country"]):
        return PROMPT.format(place=city["place"])
    return PROMPT_WITH_COUNTRY.format(place=city["place"], country=city["country"])


def write_split(path: Path, rows: list[dict], extra_fields: list[str]) -> None:
    fields = ["sentence", "country", "latitude", "longitude", "geonameid", "place", "iso"] + extra_fields
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_world_copy(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sentence", "country", "latitude", "longitude"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    cities_path = RAW_DIR / "cities15000.txt"
    country_path = RAW_DIR / "countryInfo.txt"
    for path in (cities_path, country_path):
        if not path.exists():
            raise SystemExit(f"missing raw file: {path} (see module docstring for URLs)")

    official = parse_country_info(country_path)
    cities = read_cities(cities_path, official)
    by_country: dict[str, list[dict]] = defaultdict(list)
    for city in cities:
        by_country[city["iso"]].append(city)
    print(f"cities15000: {len(cities)} populated places, {len(by_country)} country codes")

    rng = random.Random(SEED)
    # 采样框：城市数最多的 235 个国家码（当前 dump 共 244 个，剔除 9 个单城市代码），
    # 使国家覆盖数与指南2 的 235 对齐；随后对每个国家至少抽 1 条，避免小国被均匀
    # 抽样漏掉。
    frame_isos = sorted(by_country, key=lambda iso: (-len(by_country[iso]), iso))[:235]
    candidates = sorted(
        iso
        for iso in frame_isos
        if UNSEEN_MIN_CITIES <= len(by_country[iso]) <= UNSEEN_MAX_CITIES
    )
    unseen_isos = sorted(rng.sample(candidates, UNSEEN_COUNTRIES))
    unseen_pool = [city for iso in unseen_isos for city in by_country[iso]]
    if len(unseen_pool) < UNSEEN_N:
        raise SystemExit(f"unseen pool too small: {len(unseen_pool)}")
    unseen = rng.sample(unseen_pool, UNSEEN_N - UNSEEN_COUNTRIES)
    unseen += [rng.choice(by_country[iso]) for iso in unseen_isos]

    seen_isos = [iso for iso in frame_isos if iso not in set(unseen_isos)]
    seen_pool = [city for iso in seen_isos for city in by_country[iso]]
    in_dist = [rng.choice(by_country[iso]) for iso in seen_isos]
    chosen_ids = {city["geonameid"] for city in in_dist}
    rest_pool = [city for city in seen_pool if city["geonameid"] not in chosen_ids]
    in_dist += rng.sample(rest_pool, TRAIN_N + VALID_N + TEST_N - len(in_dist))
    rng.shuffle(in_dist)
    train = in_dist[:TRAIN_N]
    valid = in_dist[TRAIN_N : TRAIN_N + VALID_N]
    test = in_dist[TRAIN_N + VALID_N :]

    world_train = world_rows(ROOT / "datasets" / "world" / "train.csv", build_country_iso(official))
    world_valid = world_rows(ROOT / "datasets" / "world" / "valid.csv", build_country_iso(official))
    valid_names = {(name_key(row["place"]), iso) for row in world_valid for iso in row["iso"]}
    train_names = {(name_key(row["place"]), iso) for row in world_train for iso in row["iso"]}

    def mark(rows: list[dict]) -> list[dict]:
        marked = []
        for city in rows:
            key = (name_key(city["place"]), city["iso"])
            marked.append(
                dict(
                    city,
                    sentence=prompt_for(city),
                    match_orig_valid=key in valid_names,
                    match_orig_train=key in train_names,
                )
            )
        return marked

    train, valid, test, unseen = (mark(group) for group in (train, valid, test, unseen))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_split(OUT_DIR / "train.csv", train, ["match_orig_valid", "match_orig_train"])
    write_split(OUT_DIR / "valid.csv", valid, ["match_orig_valid", "match_orig_train"])
    write_split(OUT_DIR / "test.csv", test, ["match_orig_valid", "match_orig_train"])
    write_split(OUT_DIR / "test_unseen.csv", unseen, ["match_orig_valid", "match_orig_train"])
    write_world_copy(OUT_DIR / "orig_train.csv", world_train)
    write_world_copy(OUT_DIR / "orig_valid.csv", world_valid)

    sampled = train + valid + test + unseen
    sampled_isos = Counter(city["iso"] for city in sampled)
    meta = {
        "seed": SEED,
        "source": {
            "cities15000.txt": {"md5": md5(cities_path), "rows": len(cities)},
            "countryInfo.txt": {"md5": md5(country_path), "countries": len(official)},
        },
        "protocol": {
            "total": TOTAL,
            "train": TRAIN_N,
            "valid": VALID_N,
            "test": TEST_N,
            "test_unseen": UNSEEN_N,
            "unseen_countries": UNSEEN_COUNTRIES,
            "unseen_country_city_range": [UNSEEN_MIN_CITIES, UNSEEN_MAX_CITIES],
            "country_prompt": PROMPT_WITH_COUNTRY,
            "city_state_prompt": PROMPT,
        },
        "counts": {
            "sampled_rows": len(sampled),
            "distinct_countries": len(sampled_isos),
            "train_countries": len({city["iso"] for city in train}),
            "valid_countries": len({city["iso"] for city in valid}),
            "test_countries": len({city["iso"] for city in test}),
            "unseen_countries": len(unseen_isos),
        },
        "unseen_iso": unseen_isos,
        "unseen_country_names": [display_country(iso, official) for iso in unseen_isos],
        "overlap": {
            "train_match_orig_valid": sum(city["match_orig_valid"] for city in train),
            "train_match_orig_train": sum(city["match_orig_train"] for city in train),
            "valid_match_orig_valid": sum(city["match_orig_valid"] for city in valid),
            "test_match_orig_valid": sum(city["match_orig_valid"] for city in test),
            "unseen_match_orig_valid": sum(city["match_orig_valid"] for city in unseen),
            "orig_valid_rows_with_iso": sum(bool(row["iso"]) for row in world_valid),
            "orig_train_rows_with_iso": sum(bool(row["iso"]) for row in world_train),
        },
        "note": (
            "cities15000 当前版本有 244 个国家码；采样框取城市数最多的 235 个"
            "（剔除 9 个单城市代码），与指南2 的 235 个代码对齐。"
            "每个采样国家至少 1 条，因此 counts.distinct_countries 应为 235。"
        ),
    }
    (DATA_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    audit = {
        "orig_valid": len(world_valid),
        "orig_valid_with_iso": meta["overlap"]["orig_valid_rows_with_iso"],
        "orig_train_with_iso": meta["overlap"]["orig_train_rows_with_iso"],
        "train_matches": [
            {"place": city["place"], "country": city["country"]}
            for city in train
            if city["match_orig_valid"]
        ],
        "note": "name+ISO 精确匹配；未匹配的 landmark 名称不会进入审计。",
    }
    (DATA_DIR / "leak_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta["counts"], ensure_ascii=False, indent=2))
    print(f"overlap: {meta['overlap']}")
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
