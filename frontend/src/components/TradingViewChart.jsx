import { useEffect, useRef, memo, useState } from 'react';
import { createChart, ColorType, CandlestickSeries, LineSeries, LineStyle } from 'lightweight-charts';
import axios from 'axios';

const RSI7_COLOR = '#eab308';   // bright yellow (secondary line)
const RSI14_COLOR = '#22c55e';  // bright green (standard line)
const BULL = '#16a34a';         // bright green — up candle
const BEAR = '#dc2626';         // bright red — down candle

function TradingViewChart({ symbol, backendTicker, livePrice }) {
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const candlestickSeriesRef = useRef(null);
  const rsi7SeriesRef = useRef(null);
  const rsi14SeriesRef = useRef(null);
  const currentDataRef = useRef([]);

  // RSI toggle state — RSI 14 on by default (the standard), RSI 7 off
  const [showRsi7, setShowRsi7] = useState(false);
  const [showRsi14, setShowRsi14] = useState(true);

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

    // --- Pane 0: candles ---
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

    // Make the candle pane taller than the RSI pane (roughly 3:1)
    try {
      const panes = chart.panes();
      if (panes && panes.length >= 2) {
        panes[0].setHeight(330);
        panes[1].setHeight(110);
      }
    } catch {
      // setHeight is best-effort; safe to ignore if API not present
    }

    const handleResize = () => {
      if (chartRef.current && chartContainerRef.current) {
        chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', handleResize);

    let isMounted = true;

    const fetchData = async () => {
      try {
        // Evaluate the backend symbol accurately
        let fetchSymbol = backendTicker;
        if (!fetchSymbol) {
          fetchSymbol = symbol;
          if (symbol.startsWith('NSE:')) fetchSymbol = symbol.replace('NSE:', '') + '.NS';
          if (symbol.startsWith('BSE:')) fetchSymbol = symbol.replace('BSE:', '') + '.BO';
        }

        const API_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');
        const response = await axios.get(`${API_URL}/api/stocks/chart/${fetchSymbol}`);

        const candleData = response.data.map(d => ({
          time: d.time,
          open: d.open,
          high: d.high,
          low: d.low,
          close: d.close,
        }));

        // RSI series — filter out null/undefined values (warm-up rows)
        const rsi7Data = response.data
          .filter(d => d.rsi_7 !== null && d.rsi_7 !== undefined)
          .map(d => ({ time: d.time, value: d.rsi_7 }));
        const rsi14Data = response.data
          .filter(d => d.rsi_14 !== null && d.rsi_14 !== undefined)
          .map(d => ({ time: d.time, value: d.rsi_14 }));

        // Deduplicate candles by time to satisfy lightweight-charts
        const uniqueData = [];
        const times = new Set();
        for (const item of candleData) {
          if (!times.has(item.time)) {
            times.add(item.time);
            uniqueData.push(item);
          }
        }
        uniqueData.sort((a, b) => new Date(a.time) - new Date(b.time));

        if (!isMounted) return;

        currentDataRef.current = uniqueData;
        candlestickSeries.setData(uniqueData);
        rsi7Series.setData(rsi7Data);
        rsi14Series.setData(rsi14Data);
        chart.timeScale().fitContent();
      } catch (err) {
        console.error("Failed to fetch historical chart data", err);
      }
    };

    fetchData();

    return () => {
      isMounted = false;
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [symbol]);

  // Toggle RSI series visibility without recreating the chart
  useEffect(() => {
    if (rsi7SeriesRef.current) rsi7SeriesRef.current.applyOptions({ visible: showRsi7 });
    if (rsi14SeriesRef.current) rsi14SeriesRef.current.applyOptions({ visible: showRsi14 });

    // Collapse RSI pane when both are off
    try {
      const panes = chartRef.current?.panes();
      if (panes && panes.length >= 2) {
        if (showRsi7 || showRsi14) {
          panes[0].setHeight(330);
          panes[1].setHeight(110);
        } else {
          panes[0].setHeight(440);
          panes[1].setHeight(1);
        }
      }
    } catch {
      // best-effort
    }
  }, [showRsi7, showRsi14]);

  // Live price updater
  useEffect(() => {
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
      console.error("Error updating live price on chart:", err);
    }
  }, [livePrice]);

  const toggleBtnStyle = (active, accent) => ({
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '10px 12px',
    borderRadius: '10px',
    border: `1px solid ${active ? accent : 'var(--border)'}`,
    background: active ? `${accent}1f` : 'var(--surface)',
    color: active ? accent : 'var(--text-secondary)',
    cursor: 'pointer',
    fontWeight: 600,
    fontSize: '13px',
    transition: 'all 0.18s ease',
    width: '100%',
    justifyContent: 'flex-start',
  });

  return (
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
        </div>

        <div className="tv-indicator-footnote">
          Dashed lines mark overbought (70) / oversold (30)
        </div>
      </div>
    </div>
  );
}

export default memo(TradingViewChart);
