import { useState, useEffect } from 'react';
import axios from 'axios';
import { Search, TrendingUp, TrendingDown, BarChart2, Activity, Zap, Target, Brain, ArrowUpRight, ArrowDownRight, Minus } from 'lucide-react';
import TradingViewChart from '../components/TradingViewChart';

const getTradingViewSymbol = (ticker) => {
  if (!ticker) return 'NSE:NIFTY'; // fallback

  const indexMap = {
    '^NSEI': 'NSE:NIFTY',
    '^NSEBANK': 'NSE:BANKNIFTY',
    '^CNXIT': 'NSE:CNXIT',
    '^CNXAUTO': 'NSE:CNXAUTO',
    '^CNXPHARMA': 'NSE:CNXPHARMA',
    '^CRSLDX': 'NSE:NIFTYJR',
    '^CNXMETAL': 'NSE:CNXMETAL',
    '^CNXFMCG': 'NSE:CNXFMCG',
    '^NSEMDCP50': 'NSE:NIFTYMIDCAP50'
  };
  if (indexMap[ticker]) return indexMap[ticker];

  if (ticker.endsWith('.NS')) return 'NSE:' + ticker.replace('.NS', '');
  if (ticker.endsWith('.BO')) return 'BSE:' + ticker.replace('.BO', '');
  return 'NSE:' + ticker; // default to NSE
};

const getColorForRecommendation = (rec) => {
  if (['Strong Buy', 'Buy'].includes(rec)) return 'var(--color-buy)';
  if (['Strong Sell', 'Sell'].includes(rec)) return 'var(--color-sell)';
  return 'var(--color-hold)';
};

const getStrategyColor = (strategy) => {
  if (strategy === 'Swing Trade') return '#8b5cf6';
  if (strategy === 'Positional') return '#06b6d4';
  return '#f97316';
};

const getChangeIcon = (val) => {
  if (val > 0) return <ArrowUpRight size={16} />;
  if (val < 0) return <ArrowDownRight size={16} />;
  return <Minus size={16} />;
};

export default function Dashboard() {
  const [ticker, setTicker] = useState('');
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);
  const [error, setError] = useState('');

  const [livePriceData, setLivePriceData] = useState(null);
  const [priceFlash, setPriceFlash] = useState(null); // 'up' or 'down'

  // Poll for live price updates
  useEffect(() => {
    let intervalId;
    if (data && data.ticker) {
      // Initialize with current data
      setLivePriceData({
        ltp: data.ltp,
        day_change: data.day_change,
        day_change_pct: data.day_change_pct,
        previous_close: data.previous_close
      });
      
      intervalId = setInterval(async () => {
        try {
          const API_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
          const res = await axios.get(`${API_URL}/api/stocks/price/${data.ticker}`);
          const newPrice = res.data;
          
          setLivePriceData(prev => {
            if (prev && newPrice.ltp !== prev.ltp) {
              setPriceFlash(newPrice.ltp > prev.ltp ? 'up' : 'down');
              setTimeout(() => setPriceFlash(null), 1000);
            }
            return newPrice;
          });
        } catch (err) {
          console.error("Failed to fetch live price", err);
        }
      }, 1000); // Poll every 1 second
    }
    return () => clearInterval(intervalId);
  }, [data]);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!ticker) return;

    setLoading(true);
    setError('');
    setData(null);

    try {
      const API_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
      const response = await axios.get(`${API_URL}/api/stocks/evaluate/${ticker}`);
      setData(response.data);
    } catch (err) {
      if (err.response && err.response.status === 404) {
        setError('Not enough data found for this ticker. Please check the symbol and try again.');
      } else {
        setError('Error connecting to the backend service.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      {/* Header + Search */}
      <div style={{ textAlign: 'center', marginBottom: '40px' }}>
        <h1 className="title">AI-Powered Market Evaluation</h1>
        <p className="subtitle">Analyze Nifty 50 & Momentum Stocks with Machine Learning</p>
        
        <form onSubmit={handleSearch} style={{ display: 'flex', gap: '12px', maxWidth: '500px', margin: '0 auto' }}>
          <input 
            type="text" 
            className="input-field" 
            placeholder="Enter Ticker (e.g. RELIANCE, TCS, INFY)" 
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            style={{ marginTop: 0 }}
          />
          <button type="submit" className="btn-primary" style={{ width: 'auto', whiteSpace: 'nowrap' }} disabled={loading}>
            {loading ? 'Analyzing...' : <><Search size={18} style={{ verticalAlign: 'middle', marginRight: '8px' }}/>Analyze</>}
          </button>
        </form>
        {error && <p style={{ color: 'var(--color-sell)', marginTop: '16px' }}>{error}</p>}
        {data?.data_stale && (
          <p style={{ color: 'var(--color-hold)', marginTop: '8px', fontSize: '13px' }}>
            ⚠ Using cached data — prices may be slightly outdated
          </p>
        )}
      </div>

      {data && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>

          {/* Row 1: LTP Card + Signal Card + Strategy Card */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px' }}>

            {/* LTP Card */}
            <div className={`glass-panel ${priceFlash === 'up' ? 'flash-up' : priceFlash === 'down' ? 'flash-down' : ''}`} style={{ borderTop: `4px solid ${livePriceData?.day_change >= 0 ? 'var(--color-buy)' : 'var(--color-sell)'}`, transition: 'all 0.5s ease' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                <span className="label" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Activity size={14} /> {data.ticker}
                </span>
                <span style={{
                  fontSize: '12px',
                  padding: '3px 10px',
                  borderRadius: '20px',
                  background: 'rgba(255,255,255,0.06)',
                  color: 'var(--text-secondary)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px'
                }}>
                  <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--color-buy)', animation: 'blink 2s infinite' }}></span>
                  LIVE
                </span>
              </div>
              <div style={{ fontSize: '38px', fontWeight: '800', letterSpacing: '-0.02em', marginBottom: '4px' }}>
                ₹{livePriceData?.ltp?.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </div>
              <div style={{
                display: 'flex', alignItems: 'center', gap: '6px',
                color: livePriceData?.day_change >= 0 ? 'var(--color-buy)' : 'var(--color-sell)',
                fontSize: '15px', fontWeight: '600',
              }}>
                {getChangeIcon(livePriceData?.day_change)}
                ₹{Math.abs(livePriceData?.day_change || 0).toFixed(2)} ({livePriceData?.day_change_pct > 0 ? '+' : ''}{livePriceData?.day_change_pct}%)
              </div>
              {livePriceData?.previous_close && (
                <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '6px' }}>
                  Prev Close: ₹{livePriceData?.previous_close?.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </div>
              )}
            </div>

            {/* Signal Card */}
            <div className="glass-panel" style={{ borderTop: `4px solid ${getColorForRecommendation(data.prediction)}` }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                <span className="label" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Brain size={14} /> ML Signal
                </span>
                <span style={{
                  fontSize: '12px',
                  padding: '3px 10px',
                  borderRadius: '20px',
                  background: `${getColorForRecommendation(data.prediction)}22`,
                  color: getColorForRecommendation(data.prediction),
                  fontWeight: '600',
                }}>
                  {data.confidence}% confident
                </span>
              </div>
              <div style={{ fontSize: '34px', fontWeight: '800', color: getColorForRecommendation(data.prediction), letterSpacing: '-0.02em' }}>
                {data.prediction}
              </div>
              <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '6px' }}>
                Model Accuracy (CV): {data.model_accuracy_cv}%
              </div>
            </div>

            {/* Strategy Card */}
            <div className="glass-panel" style={{ borderTop: `4px solid ${getStrategyColor(data.trade_strategy)}` }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                <span className="label" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Target size={14} /> Trade Strategy
                </span>
              </div>
              <div style={{ fontSize: '30px', fontWeight: '800', color: getStrategyColor(data.trade_strategy), letterSpacing: '-0.02em' }}>
                {data.trade_strategy}
              </div>
              <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '8px' }}>
                {data.trade_strategy === 'Swing Trade' && 'High volatility + clear momentum — ideal for 2-10 day trades'}
                {data.trade_strategy === 'Positional' && 'Low volatility, trend-following — hold for weeks'}
                {data.trade_strategy === 'Intraday Avoid' && 'Choppy conditions — no clear edge for short-term'}
              </div>
            </div>
          </div>

          {/* Row 2: Live Chart */}
          <div className="glass-panel" style={{ padding: '24px' }}>
            <h3 className="section-heading">
              <Activity size={16} /> Live Chart Analysis
            </h3>
            <TradingViewChart symbol={getTradingViewSymbol(data.ticker)} backendTicker={data.ticker} livePrice={livePriceData} />
          </div>

          {/* Row 3: Probability Distribution + Top Features */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '20px' }}>

            {/* Probability Distribution */}
            <div className="glass-panel">
              <h3 className="section-heading">
                <BarChart2 size={16} /> Signal Probabilities
              </h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell'].map((label) => {
                  const prob = data.all_probabilities?.[label] || 0;
                  return (
                    <div key={label}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                        <span style={{ fontSize: '13px', color: label === data.prediction ? 'white' : 'var(--text-secondary)' }}>{label}</span>
                        <span style={{ fontSize: '13px', fontWeight: '700', color: getColorForRecommendation(label) }}>{prob}%</span>
                      </div>
                      <div style={{
                        height: '6px',
                        borderRadius: '3px',
                        background: 'rgba(255,255,255,0.06)',
                        overflow: 'hidden',
                      }}>
                        <div style={{
                          height: '100%',
                          width: `${prob}%`,
                          borderRadius: '3px',
                          background: getColorForRecommendation(label),
                          transition: 'width 0.6s ease',
                        }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Top Features */}
            <div className="glass-panel">
              <h3 className="section-heading">
                <Zap size={16} /> Top Features Driving Prediction
              </h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {data.top_features?.map((f, i) => (
                  <div key={f.feature} style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span style={{
                      fontSize: '11px',
                      fontWeight: '700',
                      color: 'var(--text-secondary)',
                      minWidth: '20px',
                      textAlign: 'right',
                    }}>
                      {i + 1}.
                    </span>
                    <div style={{ flex: 1 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '3px' }}>
                        <span style={{ fontSize: '13px', fontWeight: '500' }}>{f.feature.replace(/_/g, ' ')}</span>
                        <span style={{ fontSize: '12px', color: 'var(--accent-color)', fontWeight: '600' }}>
                          {(f.importance * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div style={{
                        height: '4px',
                        borderRadius: '2px',
                        background: 'rgba(255,255,255,0.06)',
                        overflow: 'hidden',
                      }}>
                        <div style={{
                          height: '100%',
                          width: `${f.importance * 100 / (data.top_features[0]?.importance || 1)}%`,
                          borderRadius: '2px',
                          background: `linear-gradient(90deg, var(--accent-color), #8b5cf6)`,
                          transition: 'width 0.6s ease',
                        }} />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Row 3: Full Technicals */}
          <div className="glass-panel">
            <h3 className="section-heading">
              <TrendingUp size={16} /> Technical Indicators
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
              {[
                { label: 'SMA 50', value: data.technicals.SMA_50, prefix: '₹' },
                { label: 'SMA 200', value: data.technicals.SMA_200, prefix: '₹' },
                { label: 'RSI (14)', value: data.technicals.RSI_14, color: data.technicals.RSI_14 > 70 ? 'var(--color-sell)' : data.technicals.RSI_14 < 30 ? 'var(--color-buy)' : 'white' },
                { label: 'MACD', value: data.technicals.MACD, color: data.technicals.MACD > 0 ? 'var(--color-buy)' : 'var(--color-sell)' },
                { label: 'ADX', value: data.technicals.ADX },
                { label: 'ATR (14)', value: data.technicals.ATR_14 },
                { label: 'Bollinger Width', value: data.technicals.BB_Width },
                { label: 'Stochastic %K', value: data.technicals.Stoch_K },
                { label: 'OBV', value: data.technicals.OBV?.toLocaleString('en-IN') },
              ].map((item) => (
                <div key={item.label} style={{
                  padding: '14px',
                  borderRadius: '10px',
                  background: 'rgba(255,255,255,0.03)',
                  border: '1px solid rgba(255,255,255,0.05)',
                }}>
                  <div className="label" style={{ fontSize: '12px', marginBottom: '4px' }}>{item.label}</div>
                  <div style={{ fontSize: '18px', fontWeight: '700', color: item.color || 'white' }}>
                    {item.prefix || ''}{item.value}
                  </div>
                </div>
              ))}
            </div>
          </div>

        </div>
      )}
    </div>
  );
}
