from ml_engine import StockMLEngine
import json
engine = StockMLEngine()
try:
    res = engine.evaluate('RELIANCE.NS')
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
