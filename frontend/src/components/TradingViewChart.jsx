import { useEffect, useRef, memo, useState } from 'react';
import {
  createChart,
  ColorType,
  CandlestickSeries,
  LineSeries,
  HistogramSeries,
  LineStyle,
} from 'lightweight-charts';
import axios from 'axios';

const RSI7_COLOR = '#eab308';   // bright yellow (secondary line)
const RSI14_COLOR = '#22c55e';  // bright green (standard line)
const BULL = '#16a34a';         // bright green — up candle / up brick / +ve histogram
const BULL_FADE = '#8bd1a5';    // soft green (50% tint of BULL) — +ve MACD bar, momentum waning
const BEAR = '#dc2626';         // bright red — down candle / down brick / -ve histogram
const BEAR_FADE = '#ee9393';    // soft pink (50% tint of BEAR) — -ve MACD bar, momentum waning
const MACD_LINE = '#2563eb';    // blue — MACD line
const MACD_SIGNAL = '#f97316';  // orange — signal line

// Chart timeframes — shortest → longest. Labels shown in the selector;
// keys match the backend `interval` query param.
const INTERVALS = [
  { key: '1m', label: '1 min' },
  { key: '5m', label: '5 mins' },
  { key: '1h', label: '1 hour' },
  { key: '1d', label: '1 day' },
];

// Chart types — candlesticks (time-based, supports indicators) and Renko
// (brick-based price action; indicators don't apply to its synthetic axis).
const CHART_TYPES = [
  { key: 'candles', label: 'Candles' },
  { key: 'renko', label: 'Renko' },
];

// ---------------------------------------------------------------------------
// Renko helpers (pure functions — no chart/state dependencies)
// ---------------------------------------------------------------------------

const toSeconds = (t) => (typeof t === 'number' ? t : Math.floor(Date.parse(t) / 1000));

// ATR(14)-style brick size from the candle series. Falls back to 1% of the last
// close when there aren't enough bars or the result isn't finite. This mirrors
// TradingView's default "ATR" Renko brick sizing.
function computeBrickSize(candles) {
  const fallback = () => {
    const lastClose = candles.length ? candles[candles.length - 1].close : 1;
    return Math.max((lastClose || 1) * 0.01, 1e-6);
  };
  if (candles.length < 15) return fallback();

  const period = 14;
  const trs = [];
  for (let i = 1; i < candles.length; i++) {
    const { high, low } = candles[i];
    const prevClose = candles[i - 1].close;
    trs.push(Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose)));
  }
  const last = trs.slice(-period);
  const atr = last.reduce((a, b) => a + b, 0) / last.length;
  if (!Number.isFinite(atr) || atr <= 0) return fallback();
  return atr;
}

// Standard 2-box-reversal Renko brick builder over Close prices. Returns
// candlestick-shaped points (up brick → green, down brick → red) with strictly
// increasing numeric times so lightweight-charts accepts them even when several
// bricks form from a single source bar.
function computeRenko(candles, brickSize) {
  const bricks = [];
  if (!candles.length || !(brickSize > 0)) return bricks;

  const round2 = (v) => Math.round(v * 100) / 100;
  let prevT = -Infinity;

  const pushBrick = (time, open, close) => {
    let t = toSeconds(time);
    if (!(t > prevT)) t = prevT + 1; // force strictly increasing
    prevT = t;
    bricks.push({
      time: t,
      open: round2(open),
      high: round2(Math.max(open, close)),
      low: round2(Math.min(open, close)),
      close: round2(close),
    });
  };

  // Emit `count` bricks of one brickSize each, stepping `dir` (+1/-1) from `start`.
  const emit = (count, start, dir, time) => {
    let level = start;
    for (let i = 0; i < count; i++) {
      const close = level + dir * brickSize;
      pushBrick(time, level, close);
      level = close;
    }
    return level;
  };

  // Anchor the first brick level to the brick grid.
  let lastClose = Math.round(candles[0].close / brickSize) * brickSize;
  let direction = 0; // 0 none, 1 up, -1 down

  for (let i = 0; i < candles.length; i++) {
    const price = candles[i].close;
    const time = candles[i].time;

    if (direction >= 0 && price >= lastClose + brickSize) {
      lastClose = emit(Math.floor((price - lastClose) / brickSize), lastClose, 1, time);
      direction = 1;
    } else if (direction <= 0 && price <= lastClose - brickSize) {
      lastClose = emit(Math.floor((lastClose - price) / brickSize), lastClose, -1, time);
      direction = -1;
    } else if (direction === 1 && price <= lastClose - 2 * brickSize) {
      // Reverse up → down: new bricks start from the previous up brick's open.
      const base = lastClose - brickSize;
      lastClose = emit(Math.floor((base - price) / brickSize), base, -1, time);
      direction = -1;
    } else if (direction === -1 && price >= lastClose + 2 * brickSize) {
      // Reverse down → up.
      const base = lastClose + brickSize;
      lastClose = emit(Math.floor((price - base) / brickSize), base, 1, time);
      direction = 1;
    }
  }

  return bricks;
}

function TradingViewChart({ symbol, backendTicker, livePrice }) {
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const candlestickSeriesRef = useRef(null);
  const rsi7SeriesRef = useRef(null);
  const rsi14SeriesRef = useRef(null);
  const macdHistSeriesRef = useRef(null);
  const macdLineSeriesRef = useRef(null);
  const signalLineSeriesRef = useRef(null);

  const currentDataRef = useRef([]); // deduped + sorted candle rows {time,o,h,l,c}
  const rawRef = useRef([]);         // full parsed rows (with rsi/macd), same order

  // RSI toggle state — RSI 14 on by default (the standard), RSI 7 off
  const [showRsi7, setShowRsi7] = useState(false);
  const [showRsi14, setShowRsi14] = useState(true);
  // MACD off by default
  const [showMacd, setShowMacd] = useState(false);

  // Chart timeframe — daily by default
  const [timeframe, setTimeframe] = useState('1d');
  // Chart type — candlesticks by default
  const [chartType, setChartType] = useState('candles');

  // Bumped after every successful fetch so the render effect re-runs.
  const [dataVersion, setDataVersion] = useState(0);

  // Size the price/RSI/MACD panes based on which lower panes are active.
  // Indicators only apply in candle mode, so Renko collapses both lower panes.
  const applyPaneHeights = (type, rsi7On, rsi14On, macdOn) => {
    try {
      const panes = chartRef.current?.panes();
      if (!panes || panes.length < 3) return;
      const isCandles = type === 'candles';
      const rsiOn = isCandles && (rsi7On || rsi14On);
      const macdActive = isCandles && macdOn;

      let h0;
      let h1;
      let h2;
      if (rsiOn && macdActive) {
        h0 = 250; h1 = 100; h2 = 100;
      } else if (rsiOn) {
        h0 = 330; h1 = 119; h2 = 1;
      } else if (macdActive) {
        h0 = 330; h1 = 1; h2 = 119;
      } else {
        h0 = 448; h1 = 1; h2 = 1;
      }
      panes[0].setHeight(h0);
      panes[1].setHeight(h1);
      panes[2].setHeight(h2);
    } catch {
      // setHeight is best-effort; safe to ignore if API not present
    }
  };

  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create lightweight-charts instance
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#504442',
        fontFamily: "-apple-system, BlinkMacSystemFont, 'SF Pro Text', system-ui, sans-serif",
      },
      grid: {
        vertLines: { color: 'rgba(39, 19, 16, 0.06)' },
        horzLines: { color: 'rgba(39, 19, 16, 0.06)' },
      },
      timeScale: {
        borderColor: 'rgba(39, 19, 16, 0.12)',
        barSpacing: 10,
      },
      rightPriceScale: {
        borderColor: 'rgba(39, 19, 16, 0.12)',
      },
      crosshair: {
        mode: 0, // Normal mode
      },
      autoSize: true, // Requires container to have explicit dimensions
    });

    chartRef.current = chart;

    // --- Pane 0: candles / Renko bricks ---
    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: BULL,
      downColor: BEAR,
      borderVisible: false,
      wickUpColor: BULL,
      wickDownColor: BEAR,
    });
    candlestickSeriesRef.current = candlestickSeries;

    // --- Pane 1: RSI (paneIndex = 1) ---
    const rsi7Series = chart.addSeries(
      LineSeries,
      {
        color: RSI7_COLOR,
        lineWidth: 2,
        priceLineVisible: false,
        lastValueVisible: true,
        title: 'RSI 7',
        visible: showRsi7,
      },
      1
    );
    rsi7SeriesRef.current = rsi7Series;

    const rsi14Series = chart.addSeries(
      LineSeries,
      {
        color: RSI14_COLOR,
        lineWidth: 2,
        priceLineVisible: false,
        lastValueVisible: true,
        title: 'RSI 14',
        visible: showRsi14,
      },
      1
    );
    rsi14SeriesRef.current = rsi14Series;

    // Overbought / oversold reference lines (always shown when RSI pane is visible)
    rsi14Series.createPriceLine({
      price: 70,
      color: 'rgba(168, 64, 42, 0.5)',
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      axisLabelVisible: true,
      title: '70',
    });
    rsi14Series.createPriceLine({
      price: 30,
      color: 'rgba(88, 107, 77, 0.5)',
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      axisLabelVisible: true,
      title: '30',
    });

    // --- Pane 2: MACD (paneIndex = 2) ---
    const macdHistSeries = chart.addSeries(
      HistogramSeries,
      {
        base: 0,
        priceLineVisible: false,
        lastValueVisible: false,
        title: 'MACD Hist',
        visible: showMacd,
      },
      2
    );
    macdHistSeriesRef.current = macdHistSeries;

    const macdLineSeries = chart.addSeries(
      LineSeries,
      {
        color: MACD_LINE,
        lineWidth: 2,
        priceLineVisible: false,
        lastValueVisible: true,
        title: 'MACD',
        visible: showMacd,
      },
      2
    );
    macdLineSeriesRef.current = macdLineSeries;

    const signalLineSeries = chart.addSeries(
      LineSeries,
      {
        color: MACD_SIGNAL,
        lineWidth: 2,
        priceLineVisible: false,
        lastValueVisible: true,
        title: 'Signal',
        visible: showMacd,
      },
      2
    );
    signalLineSeriesRef.current = signalLineSeries;

    // Initial pane sizing
    applyPaneHeights(chartType, showRsi7, showRsi14, showMacd);

    const handleResize = () => {
      if (chartRef.current && chartContainerRef.current) {
        chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol]);

  // Fetch candle/RSI/MACD data whenever the symbol or interval changes. Parses
  // and dedupes once into refs, then bumps dataVersion so the render effect
  // (which knows the current chartType) draws the appropriate series.
  useEffect(() => {
    let isMounted = true;

    const fetchData = async () => {
      const candleSeries = candlestickSeriesRef.current;
      if (!candleSeries) return;
      try {
        let fetchSymbol = backendTicker;
        if (!fetchSymbol) {
          fetchSymbol = symbol;
          if (symbol.startsWith('NSE:')) fetchSymbol = symbol.replace('NSE:', '') + '.NS';
          if (symbol.startsWith('BSE:')) fetchSymbol = symbol.replace('BSE:', '') + '.BO';
        }

        const API_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');
        const response = await axios.get(`${API_URL}/api/stocks/chart/${fetchSymbol}?interval=${timeframe}`);

        // Dedupe by time and sort chronologically — `time` is a date string
        // (daily) or epoch seconds (intraday); normalize both for comparison.
        const seen = new Set();
        const rows = [];
        for (const d of response.data) {
          if (!seen.has(d.time)) {
            seen.add(d.time);
            rows.push(d);
          }
        }
        rows.sort((a, b) => toSeconds(a.time) - toSeconds(b.time));

        if (!isMounted) return;

        rawRef.current = rows;
        currentDataRef.current = rows.map((d) => ({
          time: d.time,
          open: d.open,
          high: d.high,
          low: d.low,
          close: d.close,
        }));
        setDataVersion((v) => v + 1);
      } catch (err) {
        console.error('Failed to fetch historical chart data', err);
      }
    };

    fetchData();

    return () => { isMounted = false; };
  }, [symbol, timeframe, backendTicker]);

  // Render the series for the active chart type. Runs on every fresh fetch and
  // whenever the chart type changes — Renko re-derives bricks from the cached
  // candles without a refetch.
  useEffect(() => {
    const candleSeries = candlestickSeriesRef.current;
    if (!candleSeries) return;

    const rows = rawRef.current;
    const candles = currentDataRef.current;

    if (chartType === 'renko') {
      const brickSize = computeBrickSize(candles);
      candleSeries.setData(computeRenko(candles, brickSize));
      // Indicators can't align with the synthetic brick axis — clear them so
      // they don't pollute the shared time scale.
      rsi7SeriesRef.current?.setData([]);
      rsi14SeriesRef.current?.setData([]);
      macdHistSeriesRef.current?.setData([]);
      macdLineSeriesRef.current?.setData([]);
      signalLineSeriesRef.current?.setData([]);
    } else {
      candleSeries.setData(candles);

      rsi7SeriesRef.current?.setData(
        rows.filter((d) => d.rsi_7 != null).map((d) => ({ time: d.time, value: d.rsi_7 }))
      );
      rsi14SeriesRef.current?.setData(
        rows.filter((d) => d.rsi_14 != null).map((d) => ({ time: d.time, value: d.rsi_14 }))
      );
      macdHistSeriesRef.current?.setData(
        rows
          .filter((d) => d.macd_hist != null)
          .map((d, i, arr) => {
            // 4-colour histogram (TradingView / broker convention): colour keys on
            // both the sign of the bar AND whether it's moving away from zero vs the
            // previous bar. Strong shade = momentum building; faded shade = momentum
            // waning (an early heads-up, often a bar or two before a zero-line cross).
            const up = d.macd_hist >= 0;
            const prev = i > 0 ? arr[i - 1].macd_hist : null;
            const strengthening =
              prev == null ? true : up ? d.macd_hist >= prev : d.macd_hist <= prev;
            const color = up
              ? (strengthening ? BULL : BULL_FADE)
              : (strengthening ? BEAR : BEAR_FADE);
            return { time: d.time, value: d.macd_hist, color };
          })
      );
      macdLineSeriesRef.current?.setData(
        rows.filter((d) => d.macd != null).map((d) => ({ time: d.time, value: d.macd }))
      );
      signalLineSeriesRef.current?.setData(
        rows.filter((d) => d.macd_signal != null).map((d) => ({ time: d.time, value: d.macd_signal }))
      );
    }

    chartRef.current?.timeScale().fitContent();
  }, [dataVersion, chartType]);

  // Toggle indicator series visibility without recreating the chart.
  useEffect(() => {
    rsi7SeriesRef.current?.applyOptions({ visible: showRsi7 });
    rsi14SeriesRef.current?.applyOptions({ visible: showRsi14 });
    macdHistSeriesRef.current?.applyOptions({ visible: showMacd });
    macdLineSeriesRef.current?.applyOptions({ visible: showMacd });
    signalLineSeriesRef.current?.applyOptions({ visible: showMacd });
  }, [showRsi7, showRsi14, showMacd]);

  // Resize panes whenever the active panes or chart type change.
  useEffect(() => {
    applyPaneHeights(chartType, showRsi7, showRsi14, showMacd);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showRsi7, showRsi14, showMacd, chartType, dataVersion]);

  // Live price updater — candle mode only (Renko bricks aren't ticked live).
  useEffect(() => {
    if (chartType !== 'candles') return;
    try {
      if (candlestickSeriesRef.current && livePrice && livePrice.ltp && currentDataRef.current.length > 0) {
        const dataStr = currentDataRef.current;
        const lastBar = dataStr[dataStr.length - 1];

        const updatedBar = { ...lastBar };
        updatedBar.close = livePrice.ltp;

        if (livePrice.ltp > updatedBar.high) updatedBar.high = livePrice.ltp;
        if (livePrice.ltp < updatedBar.low) updatedBar.low = livePrice.ltp;

        candlestickSeriesRef.current.update(updatedBar);
        dataStr[dataStr.length - 1] = updatedBar;
      }
    } catch (err) {
      console.error('Error updating live price on chart:', err);
    }
  }, [livePrice, chartType]);

  const indicatorsDisabled = chartType !== 'candles';

  const toggleBtnStyle = (active, accent) => ({
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '10px 12px',
    borderRadius: '10px',
    border: `1px solid ${active ? accent : 'var(--border)'}`,
    background: active ? `${accent}1f` : 'var(--surface)',
    color: active ? accent : 'var(--text-secondary)',
    cursor: indicatorsDisabled ? 'not-allowed' : 'pointer',
    opacity: indicatorsDisabled ? 0.45 : 1,
    fontWeight: 600,
    fontSize: '13px',
    transition: 'all 0.18s ease',
    width: '100%',
    justifyContent: 'flex-start',
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      {/* Timeframe + chart-type controls */}
      <div className="tv-chart-controls">
        <div className="tv-timeframe-bar">
          {INTERVALS.map(({ key, label }) => {
            const active = timeframe === key;
            return (
              <button
                key={key}
                type="button"
                onClick={() => setTimeframe(key)}
                className={`tv-timeframe-btn${active ? ' active' : ''}`}
              >
                {label}
              </button>
            );
          })}
        </div>

        <div className="tv-timeframe-bar tv-charttype-bar">
          {CHART_TYPES.map(({ key, label }) => {
            const active = chartType === key;
            return (
              <button
                key={key}
                type="button"
                onClick={() => setChartType(key)}
                className={`tv-timeframe-btn${active ? ' active' : ''}`}
                title={key === 'renko'
                  ? 'Renko — fixed-size price bricks (ATR-based), filters out time and noise'
                  : 'Candlestick chart'}
              >
                {label}
              </button>
            );
          })}
        </div>
      </div>

      <div className="tv-chart-layout">
        <div className="tv-chart-area">
          <div ref={chartContainerRef} style={{ height: '100%', width: '100%' }} />
        </div>

      {/* Indicator toggle panel — vertical column on desktop, horizontal row on mobile */}
      <div className="tv-indicator-panel">
        <div
          className="tv-indicator-label"
          style={{
            fontSize: '10px',
            color: 'var(--text-secondary)',
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
            fontWeight: 700,
            marginBottom: '2px',
          }}
        >
          Indicators
        </div>

        <div className="tv-indicator-buttons">
          <button
            type="button"
            disabled={indicatorsDisabled}
            onClick={() => setShowRsi7(v => !v)}
            style={toggleBtnStyle(showRsi7, RSI7_COLOR)}
            title="Toggle RSI 7 — short-term momentum (more sensitive)"
          >
            <span
              style={{
                display: 'inline-block',
                width: '10px',
                height: '10px',
                borderRadius: '50%',
                background: RSI7_COLOR,
                boxShadow: showRsi7 ? `0 0 8px ${RSI7_COLOR}` : 'none',
              }}
            />
            RSI 7
          </button>

          <button
            type="button"
            disabled={indicatorsDisabled}
            onClick={() => setShowRsi14(v => !v)}
            style={toggleBtnStyle(showRsi14, RSI14_COLOR)}
            title="Toggle RSI 14 — standard 14-period Relative Strength Index"
          >
            <span
              style={{
                display: 'inline-block',
                width: '10px',
                height: '10px',
                borderRadius: '50%',
                background: RSI14_COLOR,
                boxShadow: showRsi14 ? `0 0 8px ${RSI14_COLOR}` : 'none',
              }}
            />
            RSI 14
          </button>

          <button
            type="button"
            disabled={indicatorsDisabled}
            onClick={() => setShowMacd(v => !v)}
            style={toggleBtnStyle(showMacd, MACD_LINE)}
            title="Toggle MACD — histogram + MACD (12/26) and signal (9) lines"
          >
            <span
              style={{
                display: 'inline-block',
                width: '10px',
                height: '10px',
                borderRadius: '50%',
                background: MACD_LINE,
                boxShadow: showMacd ? `0 0 8px ${MACD_LINE}` : 'none',
              }}
            />
            MACD
          </button>
        </div>

        <div className="tv-indicator-footnote">
          {indicatorsDisabled
            ? 'Indicators are available in Candles view'
            : 'RSI: dashed lines mark overbought (70) / oversold (30)'}
        </div>
      </div>
      </div>
    </div>
  );
}

export default memo(TradingViewChart);
