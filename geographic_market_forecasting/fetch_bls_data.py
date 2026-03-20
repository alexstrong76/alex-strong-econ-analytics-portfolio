import requests
import pandas as pd
import time
import os

API_KEY = "5b6b23458c184198be362665f40c3547"

# 7 metro series confirmed working from your diagnostic
# plus state-level series as geographic proxies for the remaining metros
SERIES = {
    "SMU06310800000000001": "Los Angeles CA",
    "SMU04380600000000001": "Phoenix AZ",
    "SMU08197400000000001": "Denver CO",
    "SMU36356200000000001": "New York NY",
    "SMU17169800000000001": "Chicago IL",
    "SMU26198200000000001": "Detroit MI",
    "SMU13120600000000001": "Atlanta GA",
    "SMS48000000000000001": "Texas (DFW proxy)",
    "SMS06000000000000001": "California (SF proxy)",
    "SMS53000000000000001": "Washington (Seattle proxy)",
    "SMS35000000000000001": "New Mexico (Albuquerque proxy)",
    "SMS39000000000000001": "Ohio (Columbus proxy)",
    "SMS12000000000000001": "Florida (Miami proxy)",
    "SMS47000000000000001": "Tennessee (Nashville proxy)",
    "SMS41000000000000001": "Oregon (Portland proxy)",
    "SMS37000000000000001": "North Carolina (Charlotte proxy)",
    "SMS51000000000000001": "Virginia (Richmond proxy)",
    "SMS27000000000000001": "Minnesota (Minneapolis proxy)",
    "SMS25000000000000001": "Massachusetts (Boston proxy)",
    "SMS04000000000000001": "Arizona statewide",
}

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

series_ids = list(SERIES.keys())
batches = [series_ids[i:i+10] for i in range(0, len(series_ids), 10)]
records = []

for batch_num, batch in enumerate(batches, 1):
    print(f"Fetching batch {batch_num} of {len(batches)}...")
    payload = {
        "seriesid": batch,
        "startyear": "2015",
        "endyear": "2023",
        "registrationkey": API_KEY,
        "annualaverage": True,
    }
    response = requests.post(
        "https://api.bls.gov/publicAPI/v2/timeseries/data/",
        json=payload,
        headers={"Content-type": "application/json"},
    )
    data = response.json()

    if data.get("status") != "REQUEST_SUCCEEDED":
        print(f"  Warning: {data.get('message', 'unknown error')}")

    for series in data.get("Results", {}).get("series", []):
        sid = series["seriesID"]
        name = SERIES.get(sid, sid)
        annual_obs = [o for o in series["data"] if o.get("period") == "M13"]
        for obs in annual_obs:
            try:
                emp = float(obs["value"].replace(",", ""))
            except (ValueError, AttributeError):
                continue
            records.append({
                "metro": name,
                "year": int(obs["year"]),
                "employment_thousands": round(emp / 1000, 2),
            })
        print(f"  {name:<40} {len(annual_obs)} annual obs")

    time.sleep(0.5)

if not records:
    print("\nNo records returned. Check your API key.")
else:
    df = (
        pd.DataFrame(records)
        .sort_values(["metro", "year"])
        .reset_index(drop=True)
    )
    df.to_csv("data/raw/bls_metro_employment.csv", index=False)
    print(f"\nDone. Saved {len(df)} rows across {df['metro'].nunique()} metros.")
    print("\nRow counts per metro:")
    print(df.groupby("metro")["year"].count().to_string())