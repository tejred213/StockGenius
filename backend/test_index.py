import yfinance as yf
indices = {
    "NIFTY 50": "^NSEI",
    "NIFTY BANK": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
    "NIFTY AUTO": "^CNXAUTO",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY NEXT 50": "^CRSLDX",
    "NIFTY MIDCAP 50": "^NSEMDCP50",
    "NIFTY METAL": "^CNXMETAL",
    "NIFTY FMCG": "^CNXFMCG",
}
for name, ticker in indices.items():
    try:
        df = yf.Ticker(ticker).history(period="1mo")
        print(f"{name} ({ticker}): {len(df)} rows. Volume null count: {df['Volume'].isnull().sum()}")
    except Exception as e:
        print(f"{name} ({ticker}): ERROR - {e}")
