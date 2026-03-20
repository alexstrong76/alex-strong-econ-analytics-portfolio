import requests
import pandas as pd

API_KEY = "5b6b23458c184198be362665f40c3547"

METROS = {
    "SMU48194900000000001": "Dallas-Fort Worth TX",
    "SMU48264300000000001": "Houston TX",
    "SMU48411400000000001": "San Antonio TX",
    "SMU48226400000000001": "Austin TX",
    "SMU06310800000000001": "Los Angeles CA",
    "SMU06418800000000001": "San Francisco CA",
    "SMU06197200000000001": "San Diego CA",
    "SMU04380600000000001": "Phoenix AZ",
    "SMU53423400000000001": "Seattle WA",
    "SMU41389900000000001": "Portland OR",
    "SMU08197400000000001": "Denver CO",
    "SMU35292000000000001": "Albuquerque NM",
    "SMU36356200000000001": "New York NY",
    "SMU17169800000000001": "Chicago IL",
    "SMU26198200000000001": "Detroit MI",
    "SMU39178200000000001": "Columbus OH",
    "SMU12330100000000001": "Miami FL",
    "SMU12271000000000001": "Jacksonville FL",
    "SMU13120600000000001": "Atlanta GA",
    "SMU47346000000000001": "Nashville TN",
}

series_ids = list(METROS.keys())
batches = [series_ids[i:i+10] for i in range(0, len(series_ids), 10)]

for batch in batches:
    payload = {
        "seriesid": batch,
        "startyear": "2015",
        "endyear": "2024",
        "registrationkey": API_KEY,
        "annualaverage": True,
    }
    response = requests.post(
        "https://api.bls.gov/publicAPI/v2/timeseries/data/",
        json=payload,
        headers={"Content-type": "application/json"},
    )
    data = response.json()
    print(f"\nAPI status: {data.get('status')}")
    if data.get("message"):
        print(f"Messages: {data['message']}")

    for series in data.get("Results", {}).get("series", []):
        sid = series["seriesID"]
        name = METROS.get(sid, sid)
        annual = [o for o in series["data"] if o.get("period") == "M13"]
        print(f"  {name:<30} {sid}  →  {len(annual)} annual obs")
        