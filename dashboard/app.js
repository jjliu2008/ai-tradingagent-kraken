const REFRESH_MS = 5000;

async function fetchJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`${path} -> ${response.status}`);
  }
  return response.json();
}

function formatCurrency(value) {
  const num = Number(value || 0);
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(num);
}

function formatPercent(value, digits = 2) {
  const num = Number(value || 0);
  return `${(num * 100).toFixed(digits)}%`;
}

function formatSignedPercent(value, digits = 2) {
  const num = Number(value || 0);
  const prefix = num > 0 ? "+" : "";
  return `${prefix}${(num * 100).toFixed(digits)}%`;
}

function formatNumber(value, digits = 2) {
  return Number(value || 0).toFixed(digits);
}

function formatTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function slugLabel(value) {
  return String(value || "-")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (s) => s.toUpperCase());
}

function setText(id, value) {
  const node = document.getElementById(id);
  if (node) node.textContent = value;
}

function setHtml(id, value) {
  const node = document.getElementById(id);
  if (node) node.innerHTML = value;
}

function toneClass(value) {
  if (value > 0) return "value-positive";
  if (value < 0) return "value-negative";
  return "";
}

function renderTopCandidate(candidate) {
  if (!candidate) {
    return `<div class="candidate-label">Top Candidate</div><div class="candidate-empty">No active candidate</div>`;
  }
  const tags = (candidate.component_tags || [])
    .map((tag) => `<span class="tag">${tag}</span>`)
    .join("");
  return `
    <div class="candidate-label">Top Candidate</div>
    <div class="candidate-main">
      <div>
        <div class="candidate-pair">${candidate.pair}</div>
        <div class="subnote">${slugLabel(candidate.signal_type)}</div>
      </div>
      <div class="candidate-score">Score ${formatNumber(candidate.score, 2)}</div>
    </div>
    <div class="tag-list">${tags || '<span class="tag">No tags</span>'}</div>
    <div class="subnote">
      Trend ${formatPercent(candidate.gate_trend_strength || candidate.trend_strength || 0, 2)} |
      Volume ${formatNumber(candidate.volume_ratio || 0, 2)}x |
      VWAP ${formatSignedPercent(candidate.distance_from_vwap || 0, 2)}
    </div>
  `;
}

function renderChart(points) {
  const line = document.getElementById("chart-line");
  const area = document.getElementById("chart-area");
  const grid = document.getElementById("chart-grid");
  const empty = document.getElementById("chart-empty");
  const usable = Array.isArray(points) ? points : [];
  if (!line || !area || !grid || !empty) return;

  grid.innerHTML = "";
  for (let i = 1; i < 4; i += 1) {
    const y = 240 - i * 60;
    grid.innerHTML += `<line x1="0" y1="${y}" x2="640" y2="${y}"></line>`;
  }

  if (usable.length < 2) {
    line.setAttribute("d", "");
    area.setAttribute("d", "");
    empty.style.display = "block";
    return;
  }

  empty.style.display = "none";
  const values = usable.map((point) => Number(point.equity || 0));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const spread = Math.max(max - min, 1);
  const coords = usable.map((point, index) => {
    const x = (index / (usable.length - 1)) * 640;
    const y = 220 - ((Number(point.equity || 0) - min) / spread) * 180;
    return [x, y];
  });

  const linePath = coords
    .map(([x, y], index) => `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`)
    .join(" ");
  const areaPath = `${linePath} L 640 240 L 0 240 Z`;

  line.setAttribute("d", linePath);
  area.setAttribute("d", areaPath);
}

function renderRisk(utilization) {
  const container = document.getElementById("risk-bars");
  if (!container) return;
  if (!Array.isArray(utilization) || !utilization.length) {
    container.innerHTML = `<div class="empty-state">No guardrail evaluations yet.</div>`;
    return;
  }

  container.innerHTML = utilization
    .map((item) => {
      const ratio = Math.max(0, Math.min(Number(item.utilization ?? 0), 1.25));
      const pct = Math.min(ratio, 1) * 100;
      const tone = ratio >= 0.95 || item.passed === false ? "risk-danger" : ratio >= 0.75 ? "risk-warn" : "risk-ok";
      return `
        <div class="risk-row">
          <div class="risk-name">${slugLabel(item.name)}</div>
          <div class="risk-track"><div class="risk-fill ${tone}" style="width:${pct}%;"></div></div>
          <div class="risk-value">${item.utilization == null ? "-" : `${(ratio * 100).toFixed(0)}%`}</div>
        </div>
      `;
    })
    .join("");
}

function renderMarketTable(monitoring) {
  const body = document.getElementById("market-body");
  if (!body) return;
  const rows = monitoring?.market_watch?.pair_snapshots || [];
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="8" class="empty-state">No market snapshots yet.</td></tr>`;
    return;
  }
  body.innerHTML = rows
    .map((row) => {
      const candidate = row.candidate;
      return `
        <tr>
          <td><strong>${row.pair || "-"}</strong><div class="subnote">${row.session || "-"}</div></td>
          <td class="mono">${formatNumber(row.price, row.price < 1 ? 6 : 2)}</td>
          <td class="${toneClass(row.trend_strength || 0)}">${formatSignedPercent(row.trend_strength || 0, 2)}</td>
          <td>${formatNumber(row.volume_ratio || 0, 2)}x</td>
          <td class="${toneClass(row.distance_from_vwap || 0)}">${formatSignedPercent(row.distance_from_vwap || 0, 2)}</td>
          <td>${formatNumber(row.obi || 0, 3)}</td>
          <td>${formatPercent(row.spread_pct || 0, 3)}</td>
          <td>${candidate ? `<span class="pill pill-trade">${slugLabel(candidate.signal_type)}</span><div class="subnote">Score ${formatNumber(candidate.score, 2)}</div>` : `<span class="pill pill-warn">Watching</span>`}</td>
        </tr>
      `;
    })
    .join("");
}

function renderDecisions(items) {
  const body = document.getElementById("decisions-body");
  if (!body) return;
  if (!items.length) {
    body.innerHTML = `<tr><td colspan="7" class="empty-state">No decisions logged yet.</td></tr>`;
    return;
  }
  body.innerHTML = items
    .map((item) => {
      const actionTone = item.action === "TRADE" ? "pill-trade" : "pill-skip";
      const guardTone = item.guardrail_allowed === false ? "pill-danger" : item.guardrail_allowed === true ? "pill-pass" : "pill-warn";
      const guardText = item.guardrail_allowed === false ? "Blocked" : item.guardrail_allowed === true ? "Passed" : "Pending";
      return `
        <tr>
          <td>${formatTime(item.ts)}</td>
          <td><strong>${item.pair || "-"}</strong></td>
          <td><span class="pill ${actionTone}">${item.action || "-"}</span></td>
          <td>${formatNumber(item.confidence || 0, 2)}</td>
          <td><div>${slugLabel(item.signal_type || "signal")}</div><div class="subnote">Score ${formatNumber(item.signal_score || 0, 2)}</div></td>
          <td><span class="pill ${guardTone}">${guardText}</span><div class="subnote">${item.guardrail_summary || "-"}</div></td>
          <td>${(item.reason_tags || []).map((tag) => `<span class="tag">${tag}</span>`).join("") || "-"}</td>
        </tr>
      `;
    })
    .join("");
}

function renderTrades(items) {
  const body = document.getElementById("trades-body");
  if (!body) return;
  if (!items.length) {
    body.innerHTML = `<tr><td colspan="6" class="empty-state">No closed trades yet.</td></tr>`;
    return;
  }
  body.innerHTML = items
    .map((trade) => `
      <tr>
        <td><strong>${trade.pair || "-"}</strong></td>
        <td class="mono">${formatNumber(trade.entry, trade.entry < 1 ? 6 : 2)}</td>
        <td class="mono">${formatNumber(trade.exit, trade.exit < 1 ? 6 : 2)}</td>
        <td class="${toneClass(trade.pnl_pct || 0)}">${formatSignedPercent(trade.pnl_pct || 0, 2)}</td>
        <td>${slugLabel(trade.reason || "-")}</td>
        <td>${trade.ai_review || "-"}</td>
      </tr>
    `)
    .join("");
}

function renderEvents(items) {
  const list = document.getElementById("events-list");
  if (!list) return;
  if (!items.length) {
    list.innerHTML = `<div class="empty-state">No events available.</div>`;
    return;
  }
  list.innerHTML = items
    .map((event) => {
      const summary = event.summary || event.reason || event.error || event.pair || JSON.stringify(event).slice(0, 140);
      return `
        <div class="event-item">
          <div class="event-time">${formatTime(event.ts)}</div>
          <div class="event-name">${event.event || "-"}</div>
          <div class="event-copy">${summary}</div>
        </div>
      `;
    })
    .join("");
}

function renderStatus(status, monitoring, risk, decisions, trades, events) {
  setText("agent-mode", `${String(status.mode || "paper").toUpperCase()} MODE`);
  setText(
    "agent-construction",
    `${status.construction || "unknown"} | ${status.interval_minutes || "-"}m cadence`
  );
  setText("last-updated", `Updated ${formatTime(status.generated_at)}`);
  setText("cycle", status.cycle ?? "-");
  setText("current-value", formatCurrency(status.current_value));
  setText("starting-balance", `Start ${formatCurrency(status.starting_balance)}`);

  const pnlNode = document.getElementById("realized-pnl");
  if (pnlNode) {
    pnlNode.textContent = formatSignedPercent(status.realized_pnl_pct || 0, 2);
    pnlNode.className = `metric-value ${toneClass(status.realized_pnl_pct || 0)}`;
  }

  setText("open-position-count", `${status.open_position_count || 0} open positions`);
  setText("monitored-pairs", (status.monitored_pairs || []).join(", ") || "-");
  setText("interval-label", `${status.interval_minutes || "-"}m execution frame`);
  setText("rejected-count", String(status.candidate_stats?.rejected || 0));
  setText(
    "detected-count",
    `${status.candidate_stats?.detected || 0} evaluated this session`
  );

  const reasonNode = document.getElementById("no-trade-reason");
  if (reasonNode) {
    reasonNode.textContent = slugLabel(status.last_no_trade_reason || "watching_market");
    reasonNode.className = "reason-pill";
  }
  setText(
    "no-trade-summary",
    status.last_no_trade_summary || "Monitoring the market for the next valid setup."
  );

  setHtml("top-candidate-card", renderTopCandidate(status.top_candidate));
  renderChart(status.equity_curve || []);
  renderRisk(risk.utilization || []);
  renderMarketTable(monitoring);
  renderDecisions(decisions.items || []);
  renderTrades(trades.items || []);
  renderEvents(events.items || []);
}

async function refresh() {
  try {
    const [status, monitoring, risk, decisions, trades, events] = await Promise.all([
      fetchJson("/status"),
      fetchJson("/monitoring"),
      fetchJson("/risk"),
      fetchJson("/decisions?limit=8"),
      fetchJson("/trades?limit=8"),
      fetchJson("/events?limit=12"),
    ]);
    renderStatus(status, monitoring, risk, decisions, trades, events);
  } catch (error) {
    setText("agent-mode", "API OFFLINE");
    setText("no-trade-summary", `Dashboard could not load data: ${error.message}`);
  }
}

refresh();
setInterval(refresh, REFRESH_MS);
