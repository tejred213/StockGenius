import { useEffect, useRef, memo } from 'react';
import { createChart, ColorType, CandlestickSeries } from 'lightweight-charts';
import axios from 'axios';

function TradingViewChart({ symbol, backendTicker, livePrice }) {
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const candlestickSeriesRef = useRef(null);
  const currentDataRef = useRef([]);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create lightweight-charts instance
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#d1d4dc',
      },
      grid: {
        vertLines: { color: 'rgba(255, 255, 255, 0.05)' },
        horzLines: { color: 'rgba(255, 255, 255, 0.05)' },
      },
      timeScale: {
        borderColor: 'rgba(255, 255, 255, 0.1)',
        barSpacing: 10,
      },
      rightPriceScale: {
        borderColor: 'rgba(255, 255, 255, 0.1)',
      },
      crosshair: {
        mode: 0, // Normal mode
      },
      autoSize: true, // Requires container to have explicit dimensions
    });

    chartRef.current = chart;

    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#10b981',     // Tailwind emerald-500
      downColor: '#ef4444',   // Tailwind red-500
      borderVisible: false,
      wickUpColor: '#10b981',
      wickDownColor: '#ef4444',
    });
    
    candlestickSeriesRef.current = candlestickSeries;

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
        const formattedData = response.data.map(d => ({
          time: d.time,
          open: d.open,
          high: d.high,
          low: d.low,
          close: d.close,
        }));
        
        // Deduplicate and sort to guarantee lightweight-charts compliance
        const uniqueData = [];
        const times = new Set();
        for (const item of formattedData) {
          if (!times.has(item.time)) {
            times.add(item.time);
            uniqueData.push(item);
          }
        }
        uniqueData.sort((a, b) => new Date(a.time) - new Date(b.time));

        if (!isMounted) return;
        
        currentDataRef.current = uniqueData;
        candlestickSeries.setData(uniqueData);
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

  // Live price updater
  useEffect(() => {
    try {
      if (candlestickSeriesRef.current && livePrice && livePrice.ltp && currentDataRef.current.length > 0) {
        const dataStr = currentDataRef.current;
        const lastBar = dataStr[dataStr.length - 1];
        
        const updatedBar = { ...lastBar };
        updatedBar.close = livePrice.ltp;
        
        // Extend high/low if current price broke out of it
        if (livePrice.ltp > updatedBar.high) updatedBar.high = livePrice.ltp;
        if (livePrice.ltp < updatedBar.low) updatedBar.low = livePrice.ltp;

        candlestickSeriesRef.current.update(updatedBar);
        
        // Update our internal ref array as well so subsequent updates build on this
        dataStr[dataStr.length - 1] = updatedBar;
      }
    } catch (err) {
      console.error("Error updating live price on chart:", err);
    }
  }, [livePrice]);

  return (
    <div style={{ height: "450px", width: "100%", borderRadius: "12px", overflow: "hidden", border: "1px solid rgba(255,255,255,0.05)", background: "rgba(255,255,255,0.02)" }}>
      <div ref={chartContainerRef} style={{ height: "100%", width: "100%" }} />
    </div>
  );
}

export default memo(TradingViewChart);
