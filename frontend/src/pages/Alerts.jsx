import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  Bell, BellRing, Plus, Trash2, Pause, Play, RefreshCw,
  Loader, MessageCircle, Phone, Check,
} from 'lucide-react';

const API_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');
const MONO = "-apple-system, BlinkMacSystemFont, 'SF Pro Text', 'SF Pro Display', system-ui, sans-serif";
const PHONE_KEY = 'sg_alert_phone';

const SIGNALS = ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell'];

const ALERT_TYPES = [
  { id: 'price_above', label: 'Price rises above', needs: 'threshold', unit: '₹' },
  { id: 'price_below', label: 'Price falls below', needs: 'threshold', unit: '₹' },
  { id: 'pct_move',    label: 'Moves ± % today',   needs: 'threshold', unit: '%' },
  { id: 'signal',      label: 'ML signal becomes', needs: 'signal' },
];

const STATUS_STYLE = {
  active:    { bg: 'var(--buy-bg)',  color: 'var(--color-buy)',  label: 'Active' },
  triggered: { bg: 'var(--hold-bg)', color: 'var(--color-hold)', label: 'Triggered' },
  paused:    { bg: 'var(--surface-high)', color: 'var(--text-secondary)', label: 'Paused' },
};

const describe = (a) => {
  const tk = a.ticker.replace('.NS', '').replace('.BO', '');
  if (a.type === 'price_above') return `${tk} rises above ₹${a.threshold}`;
  if (a.type === 'price_below') return `${tk} falls below ₹${a.threshold}`;
  if (a.type === 'pct_move')    return `${tk} moves ±${a.threshold}% in a day`;
  if (a.type === 'signal')      return `${tk} signal becomes ${a.target_signal}`;
  return tk;
};

const isValidPhone = (v) => /^\+\d{8,15}$/.test(v.trim().replace(/\s/g, ''));

export default function Alerts() {
  const [phone, setPhone] = useState(() => localStorage.getItem(PHONE_KEY) || '');
  const [phoneInput, setPhoneInput] = useState('');
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [busyId, setBusyId] = useState(null);

  // Create-form state
  const [ticker, setTicker] = useState('');
  const [type, setType] = useState('price_above');
  const [threshold, setThreshold] = useState('');
  const [signal, setSignal] = useState('Strong Buy');
  const [note, setNote] = useState('');
  const [creating, setCreating] = useState(false);

  const activeType = ALERT_TYPES.find((t) => t.id === type);

  const loadAlerts = useCallback(async () => {
    if (!phone) return;
    setLoading(true);
    setError('');
    try {
      const res = await axios.get(`${API_URL}/api/alerts`, { params: { phone } });
      setAlerts(res.data.alerts || []);
    } catch {
      setError('Could not load your alerts. The backend may be waking up — try again shortly.');
    } finally {
      setLoading(false);
    }
  }, [phone]);

  useEffect(() => { loadAlerts(); }, [loadAlerts]);

  const savePhone = () => {
    const clean = phoneInput.trim().replace(/\s/g, '');
    if (!isValidPhone(clean)) {
      setError('Enter your WhatsApp number in international format, e.g. +919876543210');
      return;
    }
    localStorage.setItem(PHONE_KEY, clean);
    setPhone(clean);
    setError('');
  };

  const changeNumber = () => {
    setPhoneInput(phone);
    localStorage.removeItem(PHONE_KEY);
    setPhone('');
    setAlerts([]);
  };

  const createAlert = async (e) => {
    e.preventDefault();
    setError('');
    if (!ticker.trim()) { setError('Enter a stock ticker or name.'); return; }
    const body = { phone, ticker: ticker.trim(), type, note: note.trim() || undefined };
    if (activeType.needs === 'threshold') {
      const num = parseFloat(threshold);
      if (!(num > 0)) { setError('Enter a positive value for the threshold.'); return; }
      body.threshold = num;
    } else {
      body.target_signal = signal;
    }
    setCreating(true);
    try {
      await axios.post(`${API_URL}/api/alerts`, body);
      setTicker(''); setThreshold(''); setNote('');
      await loadAlerts();
    } catch (err) {
      setError(err?.response?.data?.detail
        ? (typeof err.response.data.detail === 'string' ? err.response.data.detail : 'Please check the alert fields.')
        : 'Could not create the alert.');
    } finally {
      setCreating(false);
    }
  };

  const patchStatus = async (id, status) => {
    setBusyId(id);
    try {
      await axios.patch(`${API_URL}/api/alerts/${id}`, { status });
      await loadAlerts();
    } catch { setError('Could not update the alert.'); }
    finally { setBusyId(null); }
  };

  const removeAlert = async (id) => {
    setBusyId(id);
    try {
      await axios.delete(`${API_URL}/api/alerts/${id}`);
      await loadAlerts();
    } catch { setError('Could not delete the alert.'); }
    finally { setBusyId(null); }
  };

  // ---------------- Onboarding (no phone yet) ----------------
  if (!phone) {
    return (
      <div style={{ maxWidth: '560px', margin: '0 auto' }}>
        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
          <span style={badgeStyle}>WhatsApp Alerts</span>
          <h1 className="title">Never miss a move</h1>
          <p className="subtitle">Get a WhatsApp ping when a stock hits your price or flips to a new ML signal.</p>
        </div>

        <div className="glass-panel" style={{ padding: '28px' }}>
          <div className="section-heading"><MessageCircle size={14} /> Connect WhatsApp</div>

          <ol style={{ margin: '0 0 22px', paddingLeft: '20px', color: 'var(--text-secondary)', fontSize: '14px', lineHeight: 1.7 }}>
            <li>On WhatsApp, send <b style={{ color: 'var(--espresso)' }}>join &lt;your-sandbox-code&gt;</b> to the StockGenius sandbox number.</li>
            <li>Enter the same phone number below (with country code).</li>
            <li>Create alerts — we'll message you when they fire.</li>
          </ol>

          <label className="label" htmlFor="phone">Your WhatsApp number</label>
          <input
            id="phone"
            className="input-field"
            placeholder="+91 98765 43210"
            value={phoneInput}
            onChange={(e) => setPhoneInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && savePhone()}
          />
          {error && <p style={errorStyle}>{error}</p>}
          <button className="btn-primary" style={{ marginTop: '18px' }} onClick={savePhone}>
            <Phone size={16} style={{ verticalAlign: 'middle', marginRight: '8px' }} />
            Continue
          </button>
          <p style={{ fontSize: '12px', color: 'var(--text-faint)', marginTop: '14px', textAlign: 'center' }}>
            Your number is stored only in this browser and used to address your alerts.
          </p>
        </div>
      </div>
    );
  }

  // ---------------- Main (phone set) ----------------
  return (
    <div style={{ maxWidth: '760px', margin: '0 auto' }}>
      <div style={{ textAlign: 'center', marginBottom: '32px' }}>
        <span style={badgeStyle}>WhatsApp Alerts</span>
        <h1 className="title">Your Alerts</h1>
        <p className="subtitle">
          Delivering to <b style={{ color: 'var(--espresso)' }}>{phone}</b>{' '}
          <button onClick={changeNumber} style={linkBtnStyle}>change</button>
        </p>
      </div>

      {/* Create form */}
      <div className="glass-panel" style={{ padding: '24px', marginBottom: '24px' }}>
        <div className="section-heading"><Plus size={14} /> New alert</div>
        <form onSubmit={createAlert}>
          <div style={{ display: 'flex', gap: '14px', flexWrap: 'wrap' }}>
            <div style={{ flex: '1 1 200px' }}>
              <label className="label">Stock</label>
              <input className="input-field" placeholder="RELIANCE" value={ticker}
                     onChange={(e) => setTicker(e.target.value)} />
            </div>
            <div style={{ flex: '1 1 200px' }}>
              <label className="label">Condition</label>
              <select className="input-field" value={type} onChange={(e) => setType(e.target.value)}>
                {ALERT_TYPES.map((t) => <option key={t.id} value={t.id}>{t.label}</option>)}
              </select>
            </div>
            <div style={{ flex: '1 1 160px' }}>
              {activeType.needs === 'threshold' ? (
                <>
                  <label className="label">{activeType.unit === '%' ? 'Percent' : 'Price (₹)'}</label>
                  <input className="input-field" type="number" step="any" min="0"
                         placeholder={activeType.unit === '%' ? '5' : '1400'}
                         value={threshold} onChange={(e) => setThreshold(e.target.value)} />
                </>
              ) : (
                <>
                  <label className="label">Signal</label>
                  <select className="input-field" value={signal} onChange={(e) => setSignal(e.target.value)}>
                    {SIGNALS.map((s) => <option key={s} value={s}>{s}</option>)}
                  </select>
                </>
              )}
            </div>
          </div>
          <div style={{ marginTop: '14px' }}>
            <label className="label">Note (optional)</label>
            <input className="input-field" placeholder="e.g. add on breakout" value={note}
                   onChange={(e) => setNote(e.target.value)} />
          </div>
          {error && <p style={errorStyle}>{error}</p>}
          <button className="btn-primary" style={{ marginTop: '18px', width: 'auto', minWidth: '180px' }}
                  type="submit" disabled={creating}>
            {creating
              ? <><Loader size={16} style={{ animation: 'spin 1s linear infinite', verticalAlign: 'middle', marginRight: '8px' }} />Creating…</>
              : <><Bell size={16} style={{ verticalAlign: 'middle', marginRight: '8px' }} />Create alert</>}
          </button>
        </form>
      </div>

      {/* List */}
      <div className="glass-panel" style={{ padding: '0', overflow: 'hidden' }}>
        <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
            <BellRing size={14} style={{ verticalAlign: 'middle', marginRight: '6px' }} />
            {alerts.length} alert{alerts.length === 1 ? '' : 's'}
          </span>
          <button onClick={loadAlerts} style={linkBtnStyle} disabled={loading}>
            <RefreshCw size={13} style={{ verticalAlign: 'middle', marginRight: '4px', animation: loading ? 'spin 1s linear infinite' : 'none' }} />
            Refresh
          </button>
        </div>

        {loading && (
          <div style={{ padding: '48px', textAlign: 'center', color: 'var(--text-secondary)' }}>
            <Loader size={22} style={{ animation: 'spin 1s linear infinite' }} />
          </div>
        )}

        {!loading && alerts.length === 0 && (
          <div style={{ padding: '48px', textAlign: 'center', color: 'var(--text-secondary)' }}>
            No alerts yet. Create your first one above.
          </div>
        )}

        {!loading && alerts.map((a) => {
          const st = STATUS_STYLE[a.status] || STATUS_STYLE.active;
          const busy = busyId === a.id;
          return (
            <div key={a.id} style={{ padding: '16px 20px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: '14px', flexWrap: 'wrap', opacity: busy ? 0.5 : 1 }}>
              <div style={{ flex: '1 1 260px' }}>
                <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{describe(a)}</div>
                <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '3px' }}>
                  {a.last_value ? <>Last seen: <span className="mono">{a.last_value}</span></> : 'Not checked yet'}
                  {a.note ? ` · ${a.note}` : ''}
                </div>
              </div>
              <span style={{ ...chipBase, background: st.bg, color: st.color }}>
                {a.status === 'triggered' && <Check size={12} style={{ verticalAlign: 'middle', marginRight: '4px' }} />}
                {st.label}
              </span>
              <div style={{ display: 'flex', gap: '6px' }}>
                {a.status === 'triggered' && (
                  <button style={iconBtn} title="Re-arm" onClick={() => patchStatus(a.id, 'active')} disabled={busy}>
                    <RefreshCw size={15} />
                  </button>
                )}
                {a.status === 'active' && (
                  <button style={iconBtn} title="Pause" onClick={() => patchStatus(a.id, 'paused')} disabled={busy}>
                    <Pause size={15} />
                  </button>
                )}
                {a.status === 'paused' && (
                  <button style={iconBtn} title="Resume" onClick={() => patchStatus(a.id, 'active')} disabled={busy}>
                    <Play size={15} />
                  </button>
                )}
                <button style={{ ...iconBtn, color: 'var(--color-sell)' }} title="Delete" onClick={() => removeAlert(a.id)} disabled={busy}>
                  <Trash2 size={15} />
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ---------- inline style helpers (match Screener conventions) ----------
const badgeStyle = {
  display: 'inline-block', fontFamily: MONO, fontSize: '12px', letterSpacing: '0.12em',
  textTransform: 'uppercase', color: 'var(--text-secondary)', background: 'var(--tape)',
  padding: '5px 12px', borderRadius: '4px', border: '1px solid var(--border-strong)',
  transform: 'rotate(-1deg)', marginBottom: '18px',
};
const errorStyle = { color: 'var(--color-sell)', fontSize: '13px', marginTop: '12px', marginBottom: 0 };
const linkBtnStyle = {
  background: 'none', border: 'none', color: 'var(--accent-warm)', cursor: 'pointer',
  fontSize: '13px', fontWeight: 600, padding: 0, textDecoration: 'underline',
};
const chipBase = {
  display: 'inline-block', whiteSpace: 'nowrap', fontFamily: MONO, fontSize: '11px',
  padding: '4px 10px', borderRadius: '6px', fontWeight: 600,
};
const iconBtn = {
  display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
  width: '34px', height: '34px', borderRadius: '8px', border: '1px solid var(--border)',
  background: 'var(--surface)', color: 'var(--espresso)', cursor: 'pointer',
};
