import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import Screener from './pages/Screener';
import Alerts from './pages/Alerts';

function App() {
  return (
    <Router>
      <div className="container">
        <Navbar />
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/screener" element={<Screener />} />
          <Route path="/alerts" element={<Alerts />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
