import { Link, useLocation } from 'react-router-dom';

export default function Navbar() {
  const location = useLocation();

  return (
    <nav className="navbar">
      <Link to="/" className="logo">
        <svg width="30" height="30" viewBox="0 0 40 40" fill="none" stroke="#271310" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
          <path d="M9 16h20v8a8 8 0 0 1-8 8h-4a8 8 0 0 1-8-8z" />
          <path d="M29 18h3a3.5 3.5 0 0 1 0 7h-3" />
          <path d="M15 7c-1.5 2 1.5 3 0 5M21 6c-1.5 2 1.5 3 0 5" opacity=".7" />
        </svg>
        StockGenius
      </Link>
      <div className="nav-links">
        <Link to="/" className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}>Dashboard</Link>
        <Link to="/screener" className={`nav-link ${location.pathname === '/screener' ? 'active' : ''}`}>Screener</Link>
      </div>
    </nav>
  );
}
