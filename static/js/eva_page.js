// JS that I use for the EVA page

const playerInput = document.getElementById("playerName");
const btnFind = document.getElementById("btnFindMatches");
const matchesBox = document.getElementById("matchesBox");
const reportBox = document.getElementById("reportBox");

const chatBox = document.getElementById("chatBox");
const chatLog = document.getElementById("chatLog");
const chatInput = document.getElementById("chatInput");
const btnSendChat = document.getElementById("btnSendChat");
const typing = document.getElementById("typing");

let currentMatches = [];
let currentPlayer = "";

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function renderMarkdown(md) {
  const raw = marked.parse(md || "");
  return DOMPurify.sanitize(raw, { USE_PROFILES: { html: true } });
}

function addBubble(role, content, isMarkdown = false) {
  const wrapper = document.createElement("div");
  wrapper.className = `bubble ${role}`;

  const inner = document.createElement("div");
  inner.className = "bubble-inner";

  if (isMarkdown) {
    inner.innerHTML = renderMarkdown(content);
  } else {
    inner.innerHTML = escapeHtml(content);
  }

  wrapper.appendChild(inner);
  chatLog.appendChild(wrapper);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function setLoading(btn, isLoading, label = "") {
  if (!btn) return;
  btn.disabled = isLoading;
  if (isLoading) {
    btn.dataset.oldText = btn.textContent;
    btn.textContent = label || "Loading...";
    btn.classList.add("is-loading");
  } else {
    btn.textContent = btn.dataset.oldText || btn.textContent;
    btn.classList.remove("is-loading");
  }
}

btnFind.addEventListener("click", async () => {
  const player = playerInput.value.trim();
  if (!player) return;

  currentPlayer = player;

  matchesBox.innerHTML = `<div class="hint">Loading matches for <b>${escapeHtml(
    player
  )}</b>...</div>`;
  reportBox.innerHTML = `
    <div class="empty-state">
      <div class="empty-icon">⚡</div>
      <div class="empty-text">Select a match to analyze.</div>
    </div>
  `;
  chatBox.style.display = "none";
  chatLog.innerHTML = "";

  try {
    setLoading(btnFind, true, "Searching...");
    const res = await fetch(
      `/api/matches?player=${encodeURIComponent(player)}`
    );
    const data = await res.json();

    currentMatches = data.matches || [];

    if (!currentMatches.length) {
      matchesBox.innerHTML = `<div class="hint">No matches found for <b>${escapeHtml(
        player
      )}</b>.</div>`;
      return;
    }

    let html = `<div class="matches-head">
      <div class="matches-title">Matches for <span class="accent">${escapeHtml(
        player
      )}</span></div>
      <div class="matches-sub">Click a match index to analyze</div>
    </div>`;

    html += `<div class="matches-grid">`;

    currentMatches.slice(0, 120).forEach((m) => {
      const cls = m.result === "Win" ? "match-win" : "match-loss";
      html += `
        <button class="match-btn ${cls}" data-index="${m.index}">
          <span class="match-id">#${m.index}</span>
          <span class="match-res">${escapeHtml(m.result)}</span>
        </button>
      `;
    });

    html += `</div>`;
    matchesBox.innerHTML = html;

    matchesBox.querySelectorAll("button[data-index]").forEach((btn) => {
      btn.addEventListener("click", () =>
        analyzeMatch(parseInt(btn.dataset.index, 10))
      );
    });
  } catch (e) {
    matchesBox.innerHTML = `<div class="hint">Error loading matches. Check Flask console.</div>`;
  } finally {
    setLoading(btnFind, false);
  }
});

async function analyzeMatch(matchIndex) {
  reportBox.innerHTML = `
    <div class="loading-state">
      <div class="spinner"></div>
      <div>
        <div class="loading-title">Analyzing match #${matchIndex}</div>
        <div class="loading-sub">SHAP can take a moment…</div>
      </div>
    </div>
  `;
  chatBox.style.display = "none";
  chatLog.innerHTML = "";
  chatInput.value = "";

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ match_index: matchIndex }),
    });

    const report = await res.json();
    if (report.error) {
      reportBox.innerHTML = `<div class="hint">${escapeHtml(
        report.error
      )}</div>`;
      return;
    }

    const p = report.prediction || {};
    const plots = report.plots || {};

    // Top stats influence list
    let topStatsHtml = "";
    if (report.top_statistics && report.top_statistics.length) {
      topStatsHtml += `<div class="mini-block">
        <div class="mini-title">Most influential stats (SHAP)</div>
        <ul class="mini-list">`;
      report.top_statistics.slice(0, 3).forEach((x) => {
        const sign = Number(x.shap_value) >= 0 ? "+" : "–";
        topStatsHtml += `<li>
          <b>${escapeHtml(x.statistics)}</b>
          <span class="muted">(impact ${sign}${Math.abs(
          Number(x.shap_value)
        ).toFixed(3)})</span>
        </li>`;
      });
      topStatsHtml += `</ul></div>`;
    }

    // Improve list
    let improveHtml = "";
    if (report.to_improve && report.to_improve.length) {
      improveHtml += `<div class="mini-block">
        <div class="mini-title">Top 3 stats to improve</div>
        <ul class="mini-list">`;
      report.to_improve.forEach((x) => {
        improveHtml += `<li>
          <b>${escapeHtml(x.statistics)}</b><br/>
          <span class="muted">You: ${Number(x.player_value).toFixed(
            2
          )} • Winners avg: ${Number(x.winner_avg).toFixed(2)}</span>
        </li>`;
      });
      improveHtml += `</ul></div>`;
    }

    // Strengths
    let strengthsHtml = "";
    if (report.strengths && report.strengths.length) {
      strengthsHtml += `<div class="mini-block">
        <div class="mini-title">Strengths (this match)</div>
        <div class="pill-row">`;
      report.strengths.slice(0, 18).forEach((s) => {
        strengthsHtml += `<span class="pill">${escapeHtml(s)}</span>`;
      });
      strengthsHtml += `</div></div>`;
    }

    const html = `
      <div class="report-head">
        <div>
          <div class="report-title">Match #${report.match_index}</div>
          <div class="report-sub">
            Player: <span class="accent">${escapeHtml(
              report.player_name || currentPlayer
            )}</span>
          </div>
        </div>
        <div class="scorebox">
          <div class="scoreline">
            <span class="label">Prediction</span>
            <span class="value">${escapeHtml(p.predicted || "")}</span>
          </div>
          <div class="scoreline">
            <span class="label">Probability</span>
            <span class="value">${Number(p.probability || 0).toFixed(2)}</span>
          </div>
          <div class="scoreline">
            <span class="label">Actual</span>
            <span class="value">${escapeHtml(p.real || "")}</span>
          </div>
        </div>
      </div>

      <div class="report-grid">
        ${topStatsHtml}
        ${improveHtml}
        ${strengthsHtml}
      </div>

      <div class="plots-title">Visual comparisons</div>
      <div class="plots-grid">
        <div class="plot-card">
          <div class="plot-cap">Your top influential stats</div>
          <img class="plot-img" src="data:image/png;base64,${
            plots.player_top3
          }" />
        </div>
        <div class="plot-card">
          <div class="plot-cap">Winners avg on the same stats</div>
          <img class="plot-img" src="data:image/png;base64,${
            plots.winners_top3
          }" />
        </div>
        <div class="plot-card">
          <div class="plot-cap">Your weakest stats (vs winners)</div>
          <img class="plot-img" src="data:image/png;base64,${
            plots.player_weak
          }" />
        </div>
        <div class="plot-card">
          <div class="plot-cap">Winners avg on those weak stats</div>
          <img class="plot-img" src="data:image/png;base64,${
            plots.winners_weak
          }" />
        </div>
      </div>
    `;

    reportBox.innerHTML = html;
    chatBox.style.display = "block";
    chatBox.scrollIntoView({ behavior: "smooth", block: "start" });
    chatInput.focus();

    // First message from EVA
    addBubble(
      "eva",
      `✅ **Analysis ready.** Ask me anything about this match.\n\nTry:\n- _Why did I lose this match?_\n- _What should I focus on next games?_\n- _Give me a training plan._`,
      true
    );
  } catch (e) {
    reportBox.innerHTML = `<div class="hint">Error analyzing match. Check Flask console.</div>`;
  }
}

document.addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  const q = chip.getAttribute("data-q");
  if (q) {
    chatInput.value = q;
    sendChat();
  }
});

btnSendChat.addEventListener("click", sendChat);
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendChat();
});

async function sendChat() {
  const q = chatInput.value.trim();
  if (!q) return;

  addBubble("you", q, false);
  chatInput.value = "";

  typing.style.display = "block";
  setLoading(btnSendChat, true, "Sending...");

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q }),
    });

    const data = await res.json();

    //  Making EVA answer in markdown
    addBubble("eva", data.answer || "No answer.", true);
  } catch (e) {
    addBubble(
      "eva",
      "⚠️ I couldn't reach the server. Check Flask is running and no error appears in the console.",
      true
    );
  } finally {
    typing.style.display = "none";
    setLoading(btnSendChat, false);
  }
}
