# StockGenius

AI-powered stock analysis platform for NSE (National Stock Exchange) markets. Real-time momentum tracking, ML-driven signals, and smart portfolio insights.

**Live Demo:** [stock-genius-in.vercel.app]((https://stock-genius-in.vercel.app/))

## Features

- 📊 **Real-time Stock Tracking** - NIFTY50 and broader market data with live price updates
- 🤖 **ML-Powered Signals** - Machine learning models for buy/sell recommendations
- 📈 **Technical Indicators** - Momentum, trend, and volatility analysis
- 💡 **Smart Momentum Lists** - Identify stocks with strong upward/downward momentum
- 🔍 **Small Caps Analysis** - Discover opportunities in emerging companies
- 📰 **News Integration** - Market news and sentiment analysis
- 💼 **Options Advisor** - Options trading strategies and insights
- 📱 **Mobile-Ready** - Works on iOS and Android via web

## Tech Stack

### Frontend
- **Framework:** React 19 with TypeScript
- **Build Tool:** Vite
- **Charting:** Lightweight Charts, Recharts
- **Routing:** React Router DOM v7
- **HTTP Client:** Axios
- **UI Icons:** Lucide React

### Backend
- **Runtime:** Python 3.10
- **Framework:** FastAPI
- **Server:** Uvicorn
- **Deployment:** Fly.io (Mumbai region)

## Quick Start

### Prerequisites
- Node.js 18+
- Python 3.10+
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/tejred213/StockGenius.git
cd StockGenius
```

2. **Setup Frontend**
```bash
cd frontend
npm install
npm run dev
```
Frontend runs on `http://localhost:5173`

3. **Setup Backend**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Backend API runs on `http://localhost:8000`

## Project Structure

```
StockGenius/
├── frontend/                 # React + Vite application
│   ├── src/
│   │   ├── components/      # Reusable React components
│   │   ├── pages/          # Page components
│   │   └── App.jsx
│   ├── package.json
│   └── vite.config.js
│
├── backend/                  # FastAPI server
│   ├── main.py             # API endpoints
│   ├── ml_engine.py        # ML models and predictions
│   ├── indicators.py       # Technical indicators
│   ├── options_advisor.py  # Options strategies
│   ├── nifty50.py         # NIFTY50 data handler
│   ├── cache_manager.py    # Caching logic
│   ├── requirements.txt
│   ├── Dockerfile
│   └── fly.toml           # Fly.io config
│
└── README.md
```

## API Endpoints

### Health Check
```bash
GET /api/health
```

### Stock Data
```bash
GET /api/stocks/nifty50          # Get NIFTY50 constituents
GET /api/stocks/{symbol}         # Get specific stock data
```

### ML Predictions
```bash
GET /api/signals/momentum        # Momentum-based signals
GET /api/signals/strong-buy      # Strong buy recommendations
GET /api/signals/small-caps      # Small cap opportunities
```

### Technical Analysis
```bash
GET /api/indicators/{symbol}     # Technical indicators
GET /api/charts/{symbol}         # Chart data
```

See `backend/main.py` for complete API documentation.

## Development

### Frontend Development
```bash
cd frontend
npm run dev      # Start dev server
npm run build    # Production build
npm run lint     # ESLint check
```

### Backend Development
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

## Deployment

### Deploy Backend to Fly.io

1. **Install Fly.io CLI**
```bash
curl -L https://fly.io/install.sh | sh
```

2. **Login to Fly.io**
```bash
flyctl auth login
```

3. **Deploy**
```bash
cd backend
flyctl deploy
```

### Deploy Frontend

Build the static files:
```bash
cd frontend
npm run build
```

Serve the `dist` folder via your hosting service (Vercel, Netlify, etc.)

## Configuration

### Environment Variables

**Backend** (create `.env` in backend directory):
```env
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false
```

**Frontend** (create `.env.local` in frontend directory):
```env
VITE_API_URL=http://localhost:8000
```

## Performance

- **Backend:** Fly.io shared-cpu-1x with 2GB RAM
- **Caching:** Smart caching of market data to reduce API calls
- **Response Times:** <200ms for most endpoints

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Roadmap

- [ ] Mobile app (React Native / Flutter)
- [ ] Portfolio tracking and alerts
- [ ] Advanced charting with technical patterns
- [ ] Backtesting engine
- [ ] Market sentiment analysis
- [ ] Community features

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or feedback, please open an issue on [GitHub](https://github.com/tejred213/StockGenius/issues).

## Author

**Tejas Redkar**
- GitHub: [@tejred213](https://github.com/tejred213)
- Email: redkartejas213@gmail.com

---

**Disclaimer:** StockGenius provides analysis and signals for educational purposes. Always conduct your own research and consult a financial advisor before making investment decisions.
