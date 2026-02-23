import { useState, useEffect } from "react";
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from "recharts";

// ── Synthetic demo data ────────────────────────────────────────────────────
const N_EPISODES = 100;
const N_DAYS = 250;

function generateTrainingData() {
  const data = [];
  let sharpe = -0.2, alpha = 0.5, drawdown = -0.3, totalReturn = -0.05;
  for (let i = 1; i <= N_EPISODES; i++) {
    sharpe += (0.015 + Math.random() * 0.008) * (i < 60 ? 1 : 0.3);
    sharpe = Math.min(sharpe, 1.85);
    alpha = Math.max(0.01, alpha * 0.985 + (Math.random() - 0.55) * 0.012);
    drawdown += (Math.random() - 0.3) * 0.005;
    drawdown = Math.max(-0.25, Math.min(0, drawdown));
    totalReturn += (0.002 + Math.random() * 0.003) * (i < 70 ? 1 : 0.4);
    data.push({
      episode: i,
      sharpe: +sharpe.toFixed(3),
      alpha: +alpha.toFixed(4),
      maxDrawdown: +drawdown.toFixed(3),
      totalReturn: +(totalReturn * 100).toFixed(2),
      criticLoss: +(Math.max(0.001, 0.5 * Math.exp(-i * 0.03) + Math.random() * 0.05)).toFixed(4),
      actorLoss: +(Math.max(-2, -0.5 - i * 0.015 + Math.random() * 0.3)).toFixed(3),
    });
  }
  return data;
}

function generateBacktestData() {
  let agentVal = 1.0, baselineVal = 1.0;
  const data = [];
  for (let d = 0; d < N_DAYS; d++) {
    const drift = 0.0003;
    agentVal *= 1 + drift + (Math.random() - 0.47) * 0.012;
    baselineVal *= 1 + drift * 0.85 + (Math.random() - 0.49) * 0.011;
    data.push({
      day: d,
      agent: +agentVal.toFixed(4),
      baseline: +baselineVal.toFixed(4),
    });
  }
  return data;
}

const METRICS = {
  agent: { sharpe: 1.72, sortino: 2.31, calmar: 2.84, maxDrawdown: -0.142, annReturn: 0.214, annVol: 0.124, winRate: 0.561, finalValue: 1214000 },
  baseline: { sharpe: 1.54, sortino: 1.98, calmar: 2.01, maxDrawdown: -0.191, annReturn: 0.187, annVol: 0.121, winRate: 0.531, finalValue: 1187000 },
};

const DJ30 = ["AAPL","AMGN","AXP","BA","CAT","CRM","CSCO","CVX","DIS","DOW","GS","HD","HON","IBM","INTC","JNJ","JPM","KO","MCD","MMM","MRK","MSFT","NKE","PG","TRV","UNH","V","VZ","WBA","WMT"];

function generateWeights() {
  const w = {};
  const raw = DJ30.map(() => Math.random());
  const sum = raw.reduce((a, b) => a + b, 0);
  DJ30.forEach((t, i) => { w[t] = raw[i] / sum; });
  return w;
}

// ── Sub-components ─────────────────────────────────────────────────────────

function MetricCard({ label, agent, baseline, format = v => v }) {
  const delta = agent - baseline;
  const pct = baseline !== 0 ? (delta / Math.abs(baseline)) * 100 : 0;
  const better = delta > 0 ? (label !== "maxDrawdown") : (label === "maxDrawdown");
  return (
    <div style={{ background: "#0f1923", border: "1px solid #1e3a4a", borderRadius: 8, padding: "16px 20px" }}>
      <div style={{ fontSize: 11, color: "#5a7a8a", textTransform: "uppercase", letterSpacing: 1, marginBottom: 6 }}>{label}</div>
      <div style={{ display: "flex", alignItems: "flex-end", gap: 12 }}>
        <div style={{ fontSize: 26, fontWeight: 700, color: "#e2f0f8", fontFamily: "monospace" }}>{format(agent)}</div>
        <div style={{ fontSize: 13, color: better ? "#4ade80" : "#f87171", marginBottom: 4 }}>
          {better ? "▲" : "▼"} {Math.abs(pct).toFixed(1)}%
        </div>
      </div>
      <div style={{ fontSize: 12, color: "#4a6070", marginTop: 4 }}>Baseline: {format(baseline)}</div>
    </div>
  );
}

const TABS = ["Training", "Backtest", "Weights", "HPO"];

export default function Dashboard() {
  const [tab, setTab] = useState("Training");
  const [trainingData] = useState(generateTrainingData);
  const [backtestData] = useState(generateBacktestData);
  const [weights] = useState(generateWeights);
  const [selectedMetric, setSelectedMetric] = useState("sharpe");

  const sortedWeights = Object.entries(weights).sort((a, b) => b[1] - a[1]);

  const hpoTrials = Array.from({ length: 50 }, (_, i) => ({
    trial: i + 1,
    sharpe: +(0.8 + Math.random() * 1.2 + (i > 30 ? 0.2 : 0)).toFixed(3),
    alpha: +(Math.random() * 0.5 + 0.01).toFixed(3),
  })).sort((a, b) => a.trial - b.trial);

  const bestTrial = hpoTrials.reduce((best, t) => t.sharpe > best.sharpe ? t : best, hpoTrials[0]);

  const trainingMetricOptions = [
    { key: "sharpe", label: "Sharpe Ratio", color: "#38bdf8" },
    { key: "totalReturn", label: "Episode Return (%)", color: "#4ade80" },
    { key: "maxDrawdown", label: "Max Drawdown", color: "#f87171" },
    { key: "alpha", label: "Entropy α", color: "#c084fc" },
    { key: "criticLoss", label: "Critic Loss", color: "#fb923c" },
  ];

  const activeMetricInfo = trainingMetricOptions.find(m => m.key === selectedMetric);

  return (
    <div style={{
      background: "#080e15",
      minHeight: "100vh",
      color: "#cdd9e5",
      fontFamily: "'IBM Plex Mono', 'Courier New', monospace",
      padding: "28px 32px",
    }}>
      {/* Header */}
      <div style={{ marginBottom: 28 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 6 }}>
          <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#38bdf8", boxShadow: "0 0 8px #38bdf8" }} />
          <span style={{ fontSize: 11, color: "#38bdf8", letterSpacing: 2, textTransform: "uppercase" }}>
            Deep RL Portfolio Optimization
          </span>
        </div>
        <h1 style={{ fontSize: 24, fontWeight: 800, color: "#e2f0f8", margin: 0, letterSpacing: -0.5 }}>
          SAC Agent · DJ30 · Backtest Dashboard
        </h1>
        <p style={{ fontSize: 12, color: "#4a6070", marginTop: 6, marginBottom: 0 }}>
          Soft Actor-Critic · Transaction costs 0.1% · Slippage 0.1% · Ray Tune (50 trials)
        </p>
      </div>

      {/* Metric cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 28 }}>
        <MetricCard label="Sharpe Ratio" agent={METRICS.agent.sharpe} baseline={METRICS.baseline.sharpe} format={v => v.toFixed(3)} />
        <MetricCard label="Sortino Ratio" agent={METRICS.agent.sortino} baseline={METRICS.baseline.sortino} format={v => v.toFixed(3)} />
        <MetricCard label="Ann. Return" agent={METRICS.agent.annReturn} baseline={METRICS.baseline.annReturn} format={v => (v * 100).toFixed(1) + "%"} />
        <MetricCard label="maxDrawdown" agent={METRICS.agent.maxDrawdown} baseline={METRICS.baseline.maxDrawdown} format={v => (v * 100).toFixed(1) + "%"} />
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: 4, marginBottom: 20, borderBottom: "1px solid #1e3a4a", paddingBottom: 0 }}>
        {TABS.map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              padding: "8px 18px",
              background: "none",
              border: "none",
              borderBottom: tab === t ? "2px solid #38bdf8" : "2px solid transparent",
              color: tab === t ? "#38bdf8" : "#4a6070",
              cursor: "pointer",
              fontSize: 12,
              letterSpacing: 1,
              textTransform: "uppercase",
              fontFamily: "inherit",
              transition: "color 0.15s",
            }}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Training tab */}
      {tab === "Training" && (
        <div>
          <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
            {trainingMetricOptions.map(m => (
              <button key={m.key} onClick={() => setSelectedMetric(m.key)} style={{
                padding: "5px 12px", borderRadius: 4, border: `1px solid ${selectedMetric === m.key ? m.color : "#1e3a4a"}`,
                background: selectedMetric === m.key ? m.color + "22" : "transparent",
                color: selectedMetric === m.key ? m.color : "#4a6070",
                fontSize: 11, cursor: "pointer", fontFamily: "inherit", letterSpacing: 0.5,
              }}>{m.label}</button>
            ))}
          </div>
          <div style={{ background: "#0a1520", border: "1px solid #1e3a4a", borderRadius: 8, padding: "20px 8px" }}>
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={trainingData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#0e2030" />
                <XAxis dataKey="episode" stroke="#2a4a5a" tick={{ fontSize: 11, fill: "#4a6070" }} label={{ value: "Episode", position: "insideBottom", offset: -2, fill: "#4a6070", fontSize: 11 }} />
                <YAxis stroke="#2a4a5a" tick={{ fontSize: 11, fill: "#4a6070" }} />
                <Tooltip contentStyle={{ background: "#0a1520", border: "1px solid #1e3a4a", fontSize: 12, fontFamily: "monospace" }} />
                <Line type="monotone" dataKey={selectedMetric} stroke={activeMetricInfo?.color} dot={false} strokeWidth={1.5} />
                {/* Rolling mean via manual smooth */}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Backtest tab */}
      {tab === "Backtest" && (
        <div>
          <div style={{ background: "#0a1520", border: "1px solid #1e3a4a", borderRadius: 8, padding: "20px 8px", marginBottom: 16 }}>
            <div style={{ fontSize: 12, color: "#5a7a8a", marginLeft: 16, marginBottom: 8, letterSpacing: 1, textTransform: "uppercase" }}>Normalised Portfolio Value (Test Period)</div>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={backtestData}>
                <defs>
                  <linearGradient id="agentGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#38bdf8" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#38bdf8" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="baseGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#fb923c" stopOpacity={0.2} />
                    <stop offset="95%" stopColor="#fb923c" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#0e2030" />
                <XAxis dataKey="day" stroke="#2a4a5a" tick={{ fontSize: 11, fill: "#4a6070" }} label={{ value: "Trading Day", position: "insideBottom", offset: -2, fill: "#4a6070", fontSize: 11 }} />
                <YAxis stroke="#2a4a5a" tick={{ fontSize: 11, fill: "#4a6070" }} />
                <Tooltip contentStyle={{ background: "#0a1520", border: "1px solid #1e3a4a", fontSize: 12, fontFamily: "monospace" }} />
                <Legend wrapperStyle={{ fontSize: 12, color: "#8aa" }} />
                <Area type="monotone" dataKey="agent" name="SAC Agent" stroke="#38bdf8" fill="url(#agentGrad)" strokeWidth={2} dot={false} />
                <Area type="monotone" dataKey="baseline" name="Equal-Weight" stroke="#fb923c" fill="url(#baseGrad)" strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
            {[
              ["Calmar Ratio", METRICS.agent.calmar, METRICS.baseline.calmar, v => v.toFixed(2)],
              ["Ann. Volatility", METRICS.agent.annVol, METRICS.baseline.annVol, v => (v * 100).toFixed(1) + "%"],
              ["Win Rate", METRICS.agent.winRate, METRICS.baseline.winRate, v => (v * 100).toFixed(1) + "%"],
              ["Final Value", METRICS.agent.finalValue, METRICS.baseline.finalValue, v => "$" + (v / 1000).toFixed(0) + "k"],
            ].map(([label, a, b, fmt]) => (
              <MetricCard key={label} label={label} agent={a} baseline={b} format={fmt} />
            ))}
          </div>
        </div>
      )}

      {/* Weights tab */}
      {tab === "Weights" && (
        <div style={{ background: "#0a1520", border: "1px solid #1e3a4a", borderRadius: 8, padding: "20px 16px" }}>
          <div style={{ fontSize: 12, color: "#5a7a8a", marginBottom: 16, letterSpacing: 1, textTransform: "uppercase" }}>Final Portfolio Weights</div>
          <ResponsiveContainer width="100%" height={360}>
            <BarChart data={sortedWeights.map(([t, w]) => ({ ticker: t, weight: +(w * 100).toFixed(2) }))} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#0e2030" horizontal={false} />
              <XAxis type="number" stroke="#2a4a5a" tick={{ fontSize: 10, fill: "#4a6070" }} unit="%" domain={[0, 8]} />
              <YAxis type="category" dataKey="ticker" stroke="#2a4a5a" tick={{ fontSize: 10, fill: "#8aa" }} width={46} />
              <Tooltip contentStyle={{ background: "#0a1520", border: "1px solid #1e3a4a", fontSize: 12, fontFamily: "monospace" }} formatter={v => v + "%"} />
              <Bar dataKey="weight" name="Weight %" fill="#38bdf8" radius={[0, 3, 3, 0]}
                   background={{ fill: "#0e2030", radius: [0, 3, 3, 0] }} />
            </BarChart>
          </ResponsiveContainer>
          <div style={{ fontSize: 11, color: "#4a6070", marginTop: 12 }}>
            Top 5: {sortedWeights.slice(0, 5).map(([t, w]) => `${t} ${(w * 100).toFixed(1)}%`).join("  ·  ")}
          </div>
        </div>
      )}

      {/* HPO tab */}
      {tab === "HPO" && (
        <div>
          <div style={{ background: "#0a1520", border: "1px solid #1e3a4a", borderRadius: 8, padding: "20px 8px", marginBottom: 16 }}>
            <div style={{ fontSize: 12, color: "#5a7a8a", marginLeft: 16, marginBottom: 8, letterSpacing: 1, textTransform: "uppercase" }}>Sharpe Ratio Across 50 Ray Tune Trials</div>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={hpoTrials}>
                <CartesianGrid strokeDasharray="3 3" stroke="#0e2030" />
                <XAxis dataKey="trial" stroke="#2a4a5a" tick={{ fontSize: 11, fill: "#4a6070" }} label={{ value: "Trial #", position: "insideBottom", offset: -2, fill: "#4a6070", fontSize: 11 }} />
                <YAxis stroke="#2a4a5a" tick={{ fontSize: 11, fill: "#4a6070" }} domain={[0.5, 2.2]} />
                <Tooltip contentStyle={{ background: "#0a1520", border: "1px solid #1e3a4a", fontSize: 12, fontFamily: "monospace" }} />
                <Line type="monotone" dataKey="sharpe" stroke="#4ade80" dot={{ r: 3, fill: "#4ade80" }} strokeWidth={1} />
                <Line type="monotone" dataKey="alpha" stroke="#c084fc" dot={false} strokeWidth={1.5} strokeDasharray="4 2" />
                <Legend wrapperStyle={{ fontSize: 12, color: "#8aa" }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div style={{ background: "#0a1520", border: "1px solid #4ade8044", borderRadius: 8, padding: "16px 20px" }}>
            <div style={{ fontSize: 11, color: "#4ade80", textTransform: "uppercase", letterSpacing: 1, marginBottom: 10 }}>
              ✓ Best Trial #{bestTrial.trial}
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16 }}>
              {[["Best Sharpe", bestTrial.sharpe.toFixed(3)], ["Entropy α", bestTrial.alpha.toFixed(3)], ["Scheduler", "ASHA"], ["Search", "HyperOpt TPE"], ["Total Trials", "50"], ["Parallelism", "4 workers"]].map(([k, v]) => (
                <div key={k}>
                  <div style={{ fontSize: 10, color: "#4a6070", letterSpacing: 1 }}>{k}</div>
                  <div style={{ fontSize: 16, fontWeight: 700, color: "#e2f0f8", fontFamily: "monospace" }}>{v}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      <div style={{ marginTop: 28, fontSize: 10, color: "#2a4a5a", borderTop: "1px solid #0e2030", paddingTop: 14 }}>
        SAC · FinRL · Ray Tune · PyTorch · DJ30 · Simulated results for demonstration
      </div>
    </div>
  );
}
