import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { Zap, BarChart2, BarChart3, Loader, Trophy, Brain } from 'lucide-react';

const API_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');

const MONO = "'JetBrains Mono', monospace";

const TABS = [
  { id: 'momentum', label: 'Momentum', icon: <Zap size={16} /> },
  { id: 'strongbuy', label: 'Strong Buy', icon: <Brain size={16} /> },
  { id: 'midcap', label: 'Mid Cap', icon: <BarChart3 size={16} /> },
  { id: 'smallcap', label: 'Small Cap', icon: <BarChart2 size={16} /> },
];

const UNIVERSE_STYLES = {
  'Nifty 50': { bg: 'rgba(101,93,90,0.14)', color: 'var(--espresso)' },
  'Mid Cap':  { bg: 'var(--buy-bg)', color: 'var(--color-buy)' },
  'Small Cap':{ bg: 'var(--hold-bg)', color: 'var(--color-hold)' },
};

const getUniverseStyle = (u) => UNIVERSE_STYLES[u] || { bg: 'var(--surface-high)', color: 'var(--text-secondary)' };

const getMomentumColor = (score) => {
  if (score >= 70) return 'var(--color-buy)';
  if (score >= 40) return 'var(--color-hold)';
  return 'var(--color-sell)';
};

const thStyle = (h) => ({
  padding: '12px 10px',
  textAlign: h === 'Stock' ? 'left' : 'center',
  color: 'var(--text-secondary)',
  fontFamily: MONO,
  fontWeight: 500,
  fontSize: '11px',
  textTransform: 'uppercase',
  letterSpacing: '0.08em',
});

const chipStyle = (u) => ({
  display: 'inline-block',
  whiteSpace: 'nowrap',
  fontFamily: MONO,
  fontSize: '11px',
  padding: '3px 9px',
  borderRadius: '6px',
  background: getUniverseStyle(u).bg,
  color: getUniverseStyle(u).color,
});

function MomentumBar({ score }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
      <div style={{ width: '60px', height: '6px', borderRadius: '4px', background: 'var(--surface-container)', overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${score}%`, borderRadius: '4px', background: getMomentumColor(score), transition: 'width 0.6s ease' }} />
      </div>
      <span className="mono" style={{ fontWeight: 700, color: getMomentumColor(score), minWidth: '36px' }}>{score}</span>
    </div>
  );
}

function MomentumTable({ stocks, onTickerClick }) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid var(--border)' }}>
            {['#', 'Stock', 'Sector', 'Universe', 'Price', 'RSI', 'MACD Hist', 'ADX', 'Momentum'].map(h => (
              <th key={h} style={thStyle(h)}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {stocks.map((s) => (
            <tr key={s.ticker} style={{ borderBottom: '1px solid var(--border)', cursor: 'pointer' }} onClick={() => onTickerClick(s.ticker)}>
              <td className="mono" style={{ padding: '12px 10px', textAlign: 'center', color: 'var(--text-secondary)', fontWeight: 700 }}>{s.rank}</td>
              <td style={{ padding: '12px 10px' }}>
                <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{s.name}</div>
                <div className="mono" style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{s.ticker.replace('.NS', '')}</div>
              </td>
              <td style={{ padding: '12px 10px', textAlign: 'center', fontSize: '12px', color: 'var(--text-secondary)' }}>{s.sector}</td>
              <td style={{ padding: '12px 10px', textAlign: 'center' }}>
                <span style={chipStyle(s.universe)}>{s.universe}</span>
              </td>
              <td className="mono" style={{ padding: '12px 10px', textAlign: 'center', fontWeight: 600, color: 'var(--espresso)' }}>₹{s.current_price?.toLocaleString('en-IN')}</td>
              <td className="mono" style={{ padding: '12px 10px', textAlign: 'center', color: s.rsi > 70 ? 'var(--color-sell)' : s.rsi < 30 ? 'var(--color-buy)' : 'var(--espresso)' }}>{s.rsi}</td>
              <td className="mono" style={{ padding: '12px 10px', textAlign: 'center', color: s.macd_histogram > 0 ? 'var(--color-buy)' : 'var(--color-sell)' }}>{s.macd_histogram?.toFixed(2)}</td>
              <td className="mono" style={{ padding: '12px 10px', textAlign: 'center', color: 'var(--espresso)' }}>{s.adx}</td>
              <td style={{ padding: '12px 10px', textAlign: 'center' }}>
                <MomentumBar score={s.momentum_score} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function StrongBuyTable({ stocks, onTickerClick }) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid var(--border)' }}>
            {['#', 'Stock', 'Sector', 'Universe', 'Price', 'Confidence', 'RSI', 'ADX', 'Momentum'].map(h => (
              <th key={h} style={thStyle(h)}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {stocks.map((s) => (
            <tr key={s.ticker} style={{ borderBottom: '1px solid var(--border)', cursor: 'pointer' }} onClick={() => onTickerClick(s.ticker)}>
              <td className="mono" style={{ padding: '12px 10px', textAlign: 'center', color: 'var(--text-secondary)', fontWeight: 700 }}>{s.rank}</td>
              <td style={{ padding: '12px 10px' }}>
                <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{s.name}</div>
                <div className="mono" style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{s.ticker.replace('.NS', '')}</div>
              </td>
              <td style={{ padding: '12px 10px', textAlign: 'center', fontSize: '12px', color: 'var(--text-secondary)' }}>{s.sector}</td>
              <td style={{ padding: '12px 10px', textAlign: 'center' }}>
                <span style={chipStyle(s.universe)}>{s.universe}</span>
              </td>
              <td className="mono" style={{ padding: '12px 10px', textAlign: 'center', fontWeight: 600, color: 'var(--espresso)' }}>₹{s.current_price?.toLocaleString('en-IN')}</td>
              <td style={{ padding: '12px 10px', textAlign: 'center' }}>
                <span className="mono" style={{ fontWeight: 700, color: 'var(--color-buy)' }}>{s.confidence}%</span>
              </td>
              <td className="mono" style={{ padding: '12px 10px', textAlign: 'center', color: s.rsi > 70 ? 'var(--color-sell)' : s.rsi < 30 ? 'var(--color-buy)' : 'var(--espresso)' }}>{s.rsi}</td>
              <td className="mono" style={{ padding: '12px 10px', textAlign: 'center', color: 'var(--espresso)' }}>{s.adx}</td>
              <td style={{ padding: '12px 10px', textAlign: 'center' }}>
                <MomentumBar score={s.momentum_score} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Screener() {
  const [activeTab, setActiveTab] = useState('momentum');
  const [momentumData, setMomentumData] = useState(null);
  const [strongBuyData, setStrongBuyData] = useState(null);
  const [midCapData, setMidCapData] = useState(null);
  const [smallCapData, setSmallCapData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleTickerClick = (ticker) => {
    const search = ticker.replace('.NS', '').replace('.BO', '');
    navigate(`/?search=${encodeURIComponent(search)}`);
  };

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError('');
      try {
        if (activeTab === 'momentum' && !momentumData) {
          const res = await axios.get(`${API_URL}/api/stocks/momentum?limit=80`);
          setMomentumData(res.data);
        } else if (activeTab === 'strongbuy' && !strongBuyData) {
          const res = await axios.get(`${API_URL}/api/stocks/strong-buys`);
          setStrongBuyData(res.data);
        } else if (activeTab === 'midcap' && !midCapData) {
          const res = await axios.get(`${API_URL}/api/stocks/midcap`);
          setMidCapData(res.data);
        } else if (activeTab === 'smallcap' && !smallCapData) {
          const res = await axios.get(`${API_URL}/api/stocks/smallcap`);
          setSmallCapData(res.data);
        }
      } catch (err) {
        setError('Failed to load data. The backend might be starting up — try again in a moment.');
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [activeTab]);

  const currentData = activeTab === 'momentum' ? momentumData
    : activeTab === 'strongbuy' ? strongBuyData
    : activeTab === 'midcap' ? midCapData
    : smallCapData;

  const stocks = activeTab === 'momentum' ? currentData?.stocks
    : activeTab === 'strongbuy' ? currentData?.stocks
    : currentData?.leaderboard;

  return (
    <div>
      <div style={{ textAlign: 'center', marginBottom: '36px' }}>
        <span style={{ display: 'inline-block', fontFamily: MONO, fontSize: '12px', letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--text-secondary)', background: 'var(--tape)', padding: '5px 12px', borderRadius: '4px', border: '1px solid var(--border-strong)', transform: 'rotate(-1deg)', marginBottom: '18px' }}>
          Today's Menu
        </span>
        <h1 className="title">Stock Screener</h1>
        <p className="subtitle">Top momentum, strong-buy signals, mid &amp; small cap opportunities — freshly brewed.</p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '24px', justifyContent: 'center', flexWrap: 'wrap' }}>
        {TABS.map(tab => {
          const active = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                display: 'flex', alignItems: 'center', gap: '8px',
                padding: '10px 20px', borderRadius: '10px',
                border: active ? '1.5px solid var(--espresso)' : '1.5px solid var(--border)',
                cursor: 'pointer', fontWeight: 600, fontSize: '14px',
                fontFamily: MONO,
                transition: 'all 0.18s ease',
                background: active ? 'var(--espresso)' : 'var(--surface)',
                color: active ? '#f3f1e9' : 'var(--text-secondary)',
                boxShadow: active ? '3px 3px 0 var(--espresso-soft)' : 'none',
              }}
            >
              {tab.icon} {tab.label}
            </button>
          );
        })}
      </div>

      {/* Content */}
      <div className="glass-panel" style={{ padding: '0', overflow: 'hidden' }}>
        {loading && (
          <div style={{ padding: '60px', textAlign: 'center', color: 'var(--text-secondary)' }}>
            <Loader size={24} style={{ animation: 'spin 1s linear infinite', marginBottom: '12px' }} />
            <p>{activeTab === 'strongbuy' ? 'Running ML evaluation on all stocks — this may take a minute...' : 'Loading stock data...'}</p>
          </div>
        )}

        {error && (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--color-sell)' }}>{error}</div>
        )}

        {!loading && !error && stocks && stocks.length > 0 && (
          <>
            <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                {activeTab === 'momentum' && <><Trophy size={14} style={{ verticalAlign: 'middle', marginRight: '6px' }} />Top {stocks.length} momentum stocks across all universes</>}
                {activeTab === 'strongbuy' && <><Brain size={14} style={{ verticalAlign: 'middle', marginRight: '6px' }} />{stocks.length} stocks with ML "Strong Buy" signal</>}
                {activeTab === 'midcap' && <><BarChart3 size={14} style={{ verticalAlign: 'middle', marginRight: '6px' }} />{stocks.length} mid cap stocks ranked by momentum</>}
                {activeTab === 'smallcap' && <><BarChart2 size={14} style={{ verticalAlign: 'middle', marginRight: '6px' }} />{stocks.length} small cap stocks ranked by momentum</>}
              </span>
              {currentData?.stale && (
                <span style={{ fontSize: '12px', color: 'var(--color-hold)' }}>Using cached data</span>
              )}
            </div>
            {activeTab === 'strongbuy'
              ? <StrongBuyTable stocks={stocks} onTickerClick={handleTickerClick} />
              : <MomentumTable stocks={stocks} onTickerClick={handleTickerClick} />
            }
          </>
        )}

        {!loading && !error && stocks && stocks.length === 0 && (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-secondary)' }}>
            {activeTab === 'strongbuy' ? 'No stocks have a "Strong Buy" signal right now.' : 'No data available.'}
          </div>
        )}
      </div>
    </div>
  );
}
