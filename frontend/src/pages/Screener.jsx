import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { TrendingUp, Zap, BarChart2, ArrowUpRight, ArrowDownRight, Minus, Loader, Trophy, Target, Brain } from 'lucide-react';

const API_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');

const TABS = [
  { id: 'momentum', label: 'Momentum', icon: <Zap size={16} /> },
  { id: 'strongbuy', label: 'Strong Buy', icon: <Brain size={16} /> },
  { id: 'smallcap', label: 'Small Cap', icon: <BarChart2 size={16} /> },
];

const getChangeIcon = (val) => {
  if (val > 0) return <ArrowUpRight size={14} />;
  if (val < 0) return <ArrowDownRight size={14} />;
  return <Minus size={14} />;
};

const getMomentumColor = (score) => {
  if (score >= 70) return 'var(--color-buy)';
  if (score >= 40) return 'var(--color-hold)';
  return 'var(--color-sell)';
};

function MomentumTable({ stocks, onTickerClick }) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
            {['#', 'Stock', 'Sector', 'Universe', 'Price', 'RSI', 'MACD Hist', 'ADX', 'Momentum'].map(h => (
              <th key={h} style={{ padding: '12px 10px', textAlign: h === 'Stock' ? 'left' : 'center', color: 'var(--text-secondary)', fontWeight: 600, fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {stocks.map((s) => (
            <tr key={s.ticker} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)', cursor: 'pointer' }} onClick={() => onTickerClick(s.ticker)}>
              <td style={{ padding: '12px 10px', textAlign: 'center', color: 'var(--text-secondary)', fontWeight: 700 }}>{s.rank}</td>
              <td style={{ padding: '12px 10px' }}>
                <div style={{ fontWeight: 600 }}>{s.name}</div>
                <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{s.ticker.replace('.NS', '')}</div>
              </td>
              <td style={{ padding: '12px 10px', textAlign: 'center', fontSize: '12px', color: 'var(--text-secondary)' }}>{s.sector}</td>
              <td style={{ padding: '12px 10px', textAlign: 'center' }}>
                <span style={{ fontSize: '11px', padding: '3px 8px', borderRadius: '12px', background: s.universe === 'Nifty 50' ? 'rgba(59,130,246,0.15)' : 'rgba(139,92,246,0.15)', color: s.universe === 'Nifty 50' ? '#60a5fa' : '#a78bfa' }}>
                  {s.universe}
                </span>
              </td>
              <td style={{ padding: '12px 10px', textAlign: 'center', fontWeight: 600 }}>₹{s.current_price?.toLocaleString('en-IN')}</td>
              <td style={{ padding: '12px 10px', textAlign: 'center', color: s.rsi > 70 ? 'var(--color-sell)' : s.rsi < 30 ? 'var(--color-buy)' : 'white' }}>{s.rsi}</td>
              <td style={{ padding: '12px 10px', textAlign: 'center', color: s.macd_histogram > 0 ? 'var(--color-buy)' : 'var(--color-sell)' }}>{s.macd_histogram?.toFixed(2)}</td>
              <td style={{ padding: '12px 10px', textAlign: 'center' }}>{s.adx}</td>
              <td style={{ padding: '12px 10px', textAlign: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                  <div style={{ width: '60px', height: '6px', borderRadius: '3px', background: 'rgba(255,255,255,0.06)', overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${s.momentum_score}%`, borderRadius: '3px', background: getMomentumColor(s.momentum_score), transition: 'width 0.6s ease' }} />
                  </div>
                  <span style={{ fontWeight: 700, color: getMomentumColor(s.momentum_score), minWidth: '36px' }}>{s.momentum_score}</span>
                </div>
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
          <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
            {['#', 'Stock', 'Sector', 'Universe', 'Price', 'Confidence', 'RSI', 'ADX', 'Momentum'].map(h => (
              <th key={h} style={{ padding: '12px 10px', textAlign: h === 'Stock' ? 'left' : 'center', color: 'var(--text-secondary)', fontWeight: 600, fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {stocks.map((s) => (
            <tr key={s.ticker} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)', cursor: 'pointer' }} onClick={() => onTickerClick(s.ticker)}>
              <td style={{ padding: '12px 10px', textAlign: 'center', color: 'var(--text-secondary)', fontWeight: 700 }}>{s.rank}</td>
              <td style={{ padding: '12px 10px' }}>
                <div style={{ fontWeight: 600 }}>{s.name}</div>
                <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{s.ticker.replace('.NS', '')}</div>
              </td>
              <td style={{ padding: '12px 10px', textAlign: 'center', fontSize: '12px', color: 'var(--text-secondary)' }}>{s.sector}</td>
              <td style={{ padding: '12px 10px', textAlign: 'center' }}>
                <span style={{ fontSize: '11px', padding: '3px 8px', borderRadius: '12px', background: s.universe === 'Nifty 50' ? 'rgba(59,130,246,0.15)' : 'rgba(139,92,246,0.15)', color: s.universe === 'Nifty 50' ? '#60a5fa' : '#a78bfa' }}>
                  {s.universe}
                </span>
              </td>
              <td style={{ padding: '12px 10px', textAlign: 'center', fontWeight: 600 }}>₹{s.current_price?.toLocaleString('en-IN')}</td>
              <td style={{ padding: '12px 10px', textAlign: 'center' }}>
                <span style={{ fontWeight: 700, color: 'var(--color-buy)' }}>{s.confidence}%</span>
              </td>
              <td style={{ padding: '12px 10px', textAlign: 'center', color: s.rsi > 70 ? 'var(--color-sell)' : s.rsi < 30 ? 'var(--color-buy)' : 'white' }}>{s.rsi}</td>
              <td style={{ padding: '12px 10px', textAlign: 'center' }}>{s.adx}</td>
              <td style={{ padding: '12px 10px', textAlign: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                  <div style={{ width: '60px', height: '6px', borderRadius: '3px', background: 'rgba(255,255,255,0.06)', overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${s.momentum_score}%`, borderRadius: '3px', background: getMomentumColor(s.momentum_score), transition: 'width 0.6s ease' }} />
                  </div>
                  <span style={{ fontWeight: 700, color: getMomentumColor(s.momentum_score), minWidth: '36px' }}>{s.momentum_score}</span>
                </div>
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
          const res = await axios.get(`${API_URL}/api/stocks/momentum?limit=30`);
          setMomentumData(res.data);
        } else if (activeTab === 'strongbuy' && !strongBuyData) {
          const res = await axios.get(`${API_URL}/api/stocks/strong-buys`);
          setStrongBuyData(res.data);
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
    : smallCapData;

  const stocks = activeTab === 'momentum' ? currentData?.stocks
    : activeTab === 'strongbuy' ? currentData?.stocks
    : currentData?.leaderboard;

  return (
    <div>
      <div style={{ textAlign: 'center', marginBottom: '32px' }}>
        <h1 className="title">Stock Screener</h1>
        <p className="subtitle">Discover top momentum stocks, strong buy signals, and small cap opportunities</p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '24px', justifyContent: 'center', flexWrap: 'wrap' }}>
        {TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              display: 'flex', alignItems: 'center', gap: '8px',
              padding: '10px 20px', borderRadius: '10px', border: 'none',
              cursor: 'pointer', fontWeight: 600, fontSize: '14px',
              transition: 'all 0.2s ease',
              background: activeTab === tab.id ? 'var(--accent-color)' : 'rgba(255,255,255,0.05)',
              color: activeTab === tab.id ? 'white' : 'var(--text-secondary)',
            }}
          >
            {tab.icon} {tab.label}
          </button>
        ))}
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
            <div style={{ padding: '16px 20px', borderBottom: '1px solid rgba(255,255,255,0.06)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                {activeTab === 'momentum' && <><Trophy size={14} style={{ verticalAlign: 'middle', marginRight: '6px' }} />Top {stocks.length} momentum stocks across all universes</>}
                {activeTab === 'strongbuy' && <><Brain size={14} style={{ verticalAlign: 'middle', marginRight: '6px' }} />{stocks.length} stocks with ML "Strong Buy" signal</>}
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
