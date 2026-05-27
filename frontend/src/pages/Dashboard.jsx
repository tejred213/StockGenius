import { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import axios from 'axios';
import { Search, TrendingUp, TrendingDown, BarChart2, Activity, Zap, Target, Brain, ArrowUpRight, ArrowDownRight, Minus, Newspaper, ExternalLink } from 'lucide-react';
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

const calculateDistancePercent = (targetPrice, currentPrice) => {
  if (!targetPrice || !currentPrice) return null;
  return (((targetPrice - currentPrice) / currentPrice) * 100).toFixed(2);
};

const formatRelativeDate = (isoDate) => {
  if (!isoDate) return '';
  const then = new Date(isoDate);
  if (Number.isNaN(then.getTime())) return '';
  const diffDays = Math.floor((Date.now() - then.getTime()) / 86400000);
  if (diffDays < 1) return 'today';
  if (diffDays === 1) return '1d ago';
  if (diffDays < 30) return `${diffDays}d ago`;
  if (diffDays < 365) return `${Math.floor(diffDays / 30)}mo ago`;
  return `${Math.floor(diffDays / 365)}y ago`;
};

const TF_LABELS = { '1h': '1H', '4h': '4H', 'daily': 'Daily' };
const TF_KEYS = ['1h', '4h', 'daily'];

export default function Dashboard() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [ticker, setTicker] = useState('');
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [newsArticles, setNewsArticles] = useState([]);
  const [newsLoading, setNewsLoading] = useState(false);
  const [selectedStrategy, setSelectedStrategy] = useState('moderate');
  const [srTimeframe, setSrTimeframe] = useState('daily');
  const [slTimeframe, setSlTimeframe] = useState('daily');

  const [livePriceData, setLivePriceData] = useState(null);
  const [priceFlash, setPriceFlash] = useState(null); // 'up' or 'down'

  // Poll for live price updates (throttled — waits for response before re-polling)
  useEffect(() => {
    let cancelled = false;
    let timeoutId;

    if (data && data.ticker) {
      // Initialize with current data
      setLivePriceData({
        ltp: data.ltp,
        day_change: data.day_change,
        day_change_pct: data.day_change_pct,
        previous_close: data.previous_close
      });

      const pollPrice = async () => {
        if (cancelled) return;
        try {
          const API_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');
          const res = await axios.get(`${API_URL}/api/stocks/price/${data.ticker}`);
          const newPrice = res.data;

          if (!cancelled) {
            setLivePriceData(prev => {
              if (prev && newPrice.ltp !== prev.ltp) {
                setPriceFlash(newPrice.ltp > prev.ltp ? 'up' : 'down');
                setTimeout(() => setPriceFlash(null), 1000);
              }
              return newPrice;
            });
          }
        } catch (err) {
          console.error("Failed to fetch live price", err);
        }
        // Schedule next poll only after current one completes
        if (!cancelled) {
          timeoutId = setTimeout(pollPrice, 5000);
        }
      };

      // Start first poll after a short delay
      timeoutId = setTimeout(pollPrice, 5000);
    }

    return () => {
      cancelled = true;
      clearTimeout(timeoutId);
    };
  }, [data]);

  const runSearch = async (searchTicker) => {
    if (!searchTicker) return;

    setLoading(true);
    setError('');
    setData(null);
    setNewsArticles([]);

    const API_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');

    try {
      const response = await axios.get(`${API_URL}/api/stocks/evaluate/${searchTicker}`);
      setData(response.data);

      // Fetch news in background
      setNewsLoading(true);
      axios.get(`${API_URL}/api/stocks/news/${searchTicker}`)
        .then(res => setNewsArticles(res.data?.articles || []))
        .catch(() => setNewsArticles([]))
        .finally(() => setNewsLoading(false));
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

  // Auto-search when navigated from Screener with ?search= param
  useEffect(() => {
    const searchQuery = searchParams.get('search');
    if (searchQuery) {
      setTicker(searchQuery.toUpperCase());
      runSearch(searchQuery);
      setSearchParams({}, { replace: true });
    }
  }, []);

  const handleSearch = async (e) => {
    e.preventDefault();
    runSearch(ticker);
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

            {/* Classical S/R (multi-timeframe) Card */}
            {data.support_resistance && (
              <div className="glass-panel" style={{ borderTop: '4px solid var(--accent-color)', gridColumn: '1 / -1' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px', flexWrap: 'wrap', gap: '12px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <TrendingUp size={14} /> <span className="label">Classical Support & Resistance</span>
                  </div>
                  {/* Timeframe tabs */}
                  <div style={{ display: 'flex', gap: '6px' }}>
                    {TF_KEYS.map((tf) => (
                      <button
                        key={tf}
                        onClick={() => setSrTimeframe(tf)}
                        style={{
                          padding: '6px 14px',
                          borderRadius: '8px',
                          border: srTimeframe === tf ? '1px solid var(--accent-color)' : '1px solid rgba(255,255,255,0.1)',
                          background: srTimeframe === tf ? 'rgba(59,130,246,0.12)' : 'rgba(255,255,255,0.03)',
                          color: srTimeframe === tf ? 'var(--accent-color)' : 'var(--text-secondary)',
                          cursor: 'pointer',
                          fontSize: '12px',
                          fontWeight: srTimeframe === tf ? '700' : '500',
                          transition: 'all 0.2s ease',
                        }}
                      >
                        {TF_LABELS[tf]}
                      </button>
                    ))}
                  </div>
                </div>

                {(() => {
                  const tfData = data.support_resistance[srTimeframe] || { resistances: [], supports: [] };
                  const hasAny = (tfData.resistances?.length || 0) + (tfData.supports?.length || 0) > 0;

                  if (!hasAny) {
                    return (
                      <div style={{ fontSize: '13px', color: 'var(--text-secondary)', padding: '12px 4px' }}>
                        Not enough {TF_LABELS[srTimeframe]} history to detect S/R levels for this stock.
                      </div>
                    );
                  }

                  return (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px' }}>
                      {/* Resistances column */}
                      <div>
                        <div style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--color-sell)', fontWeight: 700, marginBottom: '8px' }}>
                          Resistance ({tfData.resistances?.length || 0})
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                          {tfData.resistances?.length ? tfData.resistances.map((lv, i) => (
                            <div key={`r-${i}`} style={{ padding: '10px 12px', borderRadius: '8px', background: 'rgba(239,68,68,0.06)', border: '1px solid rgba(239,68,68,0.18)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '10px' }}>
                              <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', minWidth: 0 }}>
                                <div style={{ fontSize: '15px', fontWeight: 700, color: 'var(--color-sell)' }}>
                                  ₹{lv.price.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                                </div>
                                {livePriceData && (
                                  <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                                    +{calculateDistancePercent(lv.price, livePriceData.ltp)}%
                                  </div>
                                )}
                              </div>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexShrink: 0 }}>
                                <span title={`${lv.touches} touches`} style={{ padding: '2px 8px', borderRadius: '10px', background: 'rgba(255,255,255,0.06)', fontSize: '11px', fontWeight: 600, color: 'white' }}>
                                  {lv.touches}×
                                </span>
                                <span style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>
                                  {formatRelativeDate(lv.last_touched)}
                                </span>
                              </div>
                            </div>
                          )) : (
                            <div style={{ fontSize: '12px', color: 'var(--text-secondary)', padding: '8px 0' }}>
                              No resistance above current price.
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Supports column */}
                      <div>
                        <div style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--color-buy)', fontWeight: 700, marginBottom: '8px' }}>
                          Support ({tfData.supports?.length || 0})
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                          {tfData.supports?.length ? tfData.supports.map((lv, i) => (
                            <div key={`s-${i}`} style={{ padding: '10px 12px', borderRadius: '8px', background: 'rgba(16,185,129,0.06)', border: '1px solid rgba(16,185,129,0.18)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '10px' }}>
                              <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', minWidth: 0 }}>
                                <div style={{ fontSize: '15px', fontWeight: 700, color: 'var(--color-buy)' }}>
                                  ₹{lv.price.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                                </div>
                                {livePriceData && (
                                  <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                                    {calculateDistancePercent(lv.price, livePriceData.ltp)}%
                                  </div>
                                )}
                              </div>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexShrink: 0 }}>
                                <span title={`${lv.touches} touches`} style={{ padding: '2px 8px', borderRadius: '10px', background: 'rgba(255,255,255,0.06)', fontSize: '11px', fontWeight: 600, color: 'white' }}>
                                  {lv.touches}×
                                </span>
                                <span style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>
                                  {formatRelativeDate(lv.last_touched)}
                                </span>
                              </div>
                            </div>
                          )) : (
                            <div style={{ fontSize: '12px', color: 'var(--text-secondary)', padding: '8px 0' }}>
                              No support below current price.
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })()}
              </div>
            )}

            {/* StopLoss & Target Strategy Selector Card */}
            {(() => {
              const slData = data.stoploss_targets?.[slTimeframe];
              const isHold = data.prediction === 'Hold';
              const noLevels = data.stoploss_targets && !slData && !isHold;
              const borderColor = ['Buy', 'Strong Buy'].includes(data.prediction)
                ? 'var(--color-buy)'
                : ['Sell', 'Strong Sell'].includes(data.prediction)
                ? 'var(--color-sell)'
                : 'var(--text-secondary)';

              return (
                <div className="glass-panel" style={{ borderTop: `4px solid ${borderColor}`, gridColumn: '1 / -1', opacity: isHold ? 0.6 : 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px', flexWrap: 'wrap', gap: '12px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <Zap size={14} /> <span className="label">StopLoss & Target</span>
                    </div>
                    {!isHold && (
                      <div style={{ display: 'flex', gap: '6px' }}>
                        {TF_KEYS.map((tf) => (
                          <button
                            key={tf}
                            onClick={() => setSlTimeframe(tf)}
                            style={{
                              padding: '6px 14px',
                              borderRadius: '8px',
                              border: slTimeframe === tf ? '1px solid var(--accent-color)' : '1px solid rgba(255,255,255,0.1)',
                              background: slTimeframe === tf ? 'rgba(59,130,246,0.12)' : 'rgba(255,255,255,0.03)',
                              color: slTimeframe === tf ? 'var(--accent-color)' : 'var(--text-secondary)',
                              cursor: 'pointer',
                              fontSize: '12px',
                              fontWeight: slTimeframe === tf ? '700' : '500',
                              transition: 'all 0.2s ease',
                            }}
                          >
                            {TF_LABELS[tf]}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>

                  {isHold ? (
                    <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                      Not available for "Hold" signals
                    </div>
                  ) : noLevels ? (
                    <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                      Not enough {TF_LABELS[slTimeframe]} S/R levels detected for this signal direction.
                    </div>
                  ) : (
                    <>
                      {/* Strategy Selector Tabs */}
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px', marginBottom: '16px' }}>
                        {['conservative', 'moderate', 'aggressive'].map((strategy) => (
                          <button
                            key={strategy}
                            onClick={() => setSelectedStrategy(strategy)}
                            style={{
                              padding: '10px',
                              borderRadius: '8px',
                              border: selectedStrategy === strategy ? '2px solid var(--accent-color)' : '1px solid rgba(255,255,255,0.1)',
                              background: selectedStrategy === strategy ? 'rgba(59,130,246,0.10)' : 'rgba(255,255,255,0.03)',
                              color: selectedStrategy === strategy ? 'var(--accent-color)' : 'var(--text-secondary)',
                              cursor: 'pointer',
                              fontSize: '12px',
                              fontWeight: selectedStrategy === strategy ? '700' : '500',
                              transition: 'all 0.2s ease',
                            }}
                          >
                            {strategy.charAt(0).toUpperCase() + strategy.slice(1)}
                          </button>
                        ))}
                      </div>

                      {slData?.[selectedStrategy] && (
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px' }}>
                          <div style={{ padding: '14px', borderRadius: '10px', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)' }}>
                            <div className="label" style={{ fontSize: '12px', marginBottom: '8px', color: 'var(--text-secondary)' }}>
                              <ArrowDownRight size={12} style={{ display: 'inline', marginRight: '4px' }} />
                              StopLoss
                            </div>
                            <div style={{ fontSize: '18px', fontWeight: '700', color: 'var(--color-sell)', marginBottom: '4px' }}>
                              ₹{slData[selectedStrategy].stoploss?.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                            </div>
                            {livePriceData && (
                              <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                                {calculateDistancePercent(slData[selectedStrategy].stoploss, livePriceData.ltp)}%
                              </div>
                            )}
                          </div>

                          <div style={{ padding: '14px', borderRadius: '10px', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)' }}>
                            <div className="label" style={{ fontSize: '12px', marginBottom: '8px', color: 'var(--text-secondary)' }}>
                              <ArrowUpRight size={12} style={{ display: 'inline', marginRight: '4px' }} />
                              Target
                            </div>
                            <div style={{ fontSize: '18px', fontWeight: '700', color: 'var(--color-buy)', marginBottom: '4px' }}>
                              ₹{slData[selectedStrategy].target?.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                            </div>
                            {livePriceData && (
                              <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                                {calculateDistancePercent(slData[selectedStrategy].target, livePriceData.ltp)}%
                              </div>
                            )}
                          </div>

                          <div style={{ padding: '14px', borderRadius: '10px', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)' }}>
                            <div className="label" style={{ fontSize: '12px', marginBottom: '8px', color: 'var(--text-secondary)' }}>
                              Risk-Reward
                            </div>
                            <div style={{ fontSize: '20px', fontWeight: '700', color: 'var(--accent-color)' }}>
                              1 : {slData[selectedStrategy].risk_reward_ratio}
                            </div>
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </div>
              );
            })()}
          </div>

          {/* Pivot Points (secondary, intraday) */}
          {data.pivot_points && (
            <div className="glass-panel" style={{ borderTop: '4px solid rgba(255,255,255,0.15)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
                <TrendingUp size={14} /> <span className="label">Pivot Points</span>
                <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>· daily, for intraday traders</span>
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '14px' }}>
                Computed from yesterday's H/L/C using standard pivot-point formula. Reset every trading day.
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(100px, 1fr))', gap: '10px' }}>
                {[
                  { label: 'Pivot', value: data.pivot_points.pivot, color: 'var(--accent-color)' },
                  { label: 'R3', value: data.pivot_points.r3, color: 'var(--color-sell)' },
                  { label: 'R2', value: data.pivot_points.r2, color: 'var(--color-sell)' },
                  { label: 'R1', value: data.pivot_points.r1, color: 'var(--color-sell)' },
                  { label: 'S1', value: data.pivot_points.s1, color: 'var(--color-buy)' },
                  { label: 'S2', value: data.pivot_points.s2, color: 'var(--color-buy)' },
                  { label: 'S3', value: data.pivot_points.s3, color: 'var(--color-buy)' },
                ].map((item) => (
                  <div key={item.label} style={{ padding: '10px', borderRadius: '8px', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)' }}>
                    <div className="label" style={{ fontSize: '10px', color: 'var(--text-secondary)', marginBottom: '4px' }}>{item.label}</div>
                    <div style={{ fontSize: '13px', fontWeight: '700', color: item.color, marginBottom: '2px' }}>
                      ₹{item.value?.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </div>
                    {livePriceData && (
                      <div style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>
                        {calculateDistancePercent(item.value, livePriceData.ltp)}%
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

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

          {/* Row 5: News */}
          <div className="glass-panel">
            <h3 className="section-heading">
              <Newspaper size={16} /> Latest News
            </h3>
            {newsLoading && (
              <p style={{ color: 'var(--text-secondary)', fontSize: '13px' }}>Loading news...</p>
            )}
            {!newsLoading && newsArticles.length === 0 && (
              <p style={{ color: 'var(--text-secondary)', fontSize: '13px' }}>No recent news found for this stock.</p>
            )}
            {!newsLoading && newsArticles.length > 0 && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '16px' }}>
                {newsArticles.map((article, i) => (
                  <a
                    key={i}
                    href={article.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{
                      display: 'flex', gap: '12px',
                      padding: '14px', borderRadius: '12px',
                      background: 'rgba(255,255,255,0.03)',
                      border: '1px solid rgba(255,255,255,0.05)',
                      textDecoration: 'none', color: 'inherit',
                      transition: 'all 0.2s ease',
                    }}
                    onMouseEnter={e => { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.12)'; e.currentTarget.style.background = 'rgba(255,255,255,0.06)'; }}
                    onMouseLeave={e => { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.05)'; e.currentTarget.style.background = 'rgba(255,255,255,0.03)'; }}
                  >
                    {article.thumbnail && (
                      <img
                        src={article.thumbnail}
                        alt=""
                        style={{ width: '80px', height: '60px', objectFit: 'cover', borderRadius: '8px', flexShrink: 0 }}
                      />
                    )}
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: '13px', fontWeight: 600, lineHeight: 1.4, marginBottom: '6px', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                        {article.title}
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px', color: 'var(--text-secondary)' }}>
                        {article.publisher && <span>{article.publisher}</span>}
                        {article.published && <span>{new Date(article.published).toLocaleDateString('en-IN', { day: 'numeric', month: 'short' })}</span>}
                        <ExternalLink size={10} />
                      </div>
                    </div>
                  </a>
                ))}
              </div>
            )}
          </div>

        </div>
      )}
    </div>
  );
}
