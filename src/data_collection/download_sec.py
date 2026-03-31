import requests
from pathlib import Path
import time

HEADERS = {"User-Agent": "Savio savio@email.com"}

# How many filings you want
TARGET_COUNT = 20

# SEC endpoints
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

Path("../../data/raw").mkdir(parents=True, exist_ok=True)

def get_company_list():
    print("Fetching company tickers...")
    data = requests.get(TICKERS_URL, headers=HEADERS).json()
    return list(data.values())

def get_latest_10k(cik):
    cik_str = str(cik).zfill(10)
    url = SUBMISSIONS_URL.format(cik=cik_str)

    try:
        data = requests.get(url, headers=HEADERS).json()
        filings = data["filings"]["recent"]

        for i, form in enumerate(filings["form"]):
            if form == "10-K":
                accession = filings["accessionNumber"][i].replace("-", "")
                primary_doc = filings["primaryDocument"][i]
                return cik_str.lstrip("0"), accession, primary_doc
    except Exception as e:
        print(f"Error fetching CIK {cik}: {e}")

    return None

def download_filing(cik, accession, doc_name, ticker):
    url = f"{ARCHIVES_BASE}/{cik}/{accession}/{doc_name}"
    filename = f"{ticker}_{doc_name}"
    out_path = Path("../../data/raw") / filename

    print(f"Downloading {ticker} 10-K...")
    response = requests.get(url, headers=HEADERS)

    if response.status_code == 200:
        with open(out_path, "wb") as f:
            f.write(response.content)
        print(f"  Saved: {out_path} ({len(response.content)//1024} KB)")
    else:
        print(f"  Failed ({response.status_code})")

def main():
    companies = get_company_list()
    downloaded = 0

    for company in companies:
        if downloaded >= TARGET_COUNT:
            break

        cik = company["cik_str"]
        ticker = company["ticker"]

        result = get_latest_10k(cik)

        if result:
            cik_clean, accession, doc_name = result
            download_filing(cik_clean, accession, doc_name, ticker)
            downloaded += 1

            # Respect SEC rate limits
            time.sleep(0.5)

    print(f"\nDone! Downloaded {downloaded} filings.")

if __name__ == "__main__":
    main()