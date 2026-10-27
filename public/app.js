/**
 * Frontend Application Logic for Vercel.
 *
 * Supports:
 * 1. Serverless API calls to `/api/analyze`.
 * 2. High-speed client-side fallback using `model_data.json` (exact mathematical ANN forward pass).
 * 3. Dynamic category filtering and expandable card views.
 */

const SAMPLE_TEXT = `1. Account Modification and Termination
We reserve the right to amend, alter, or modify these Terms of Service and any associated fee schedules at any time in our sole discretion without prior notice to you. 
You may terminate your account at any time by contacting customer support. 
We may suspend or terminate your account and access to the investment platform immediately without notice or cause.

2. Disclaimers and Limitation of Liability
The services and financial information are provided strictly on an 'as-is' and 'as-available' basis without warranties of any kind. 
In no event shall the platform, its affiliates, or directors be liable for any direct, indirect, consequential, or punitive damages arising from market volatility or system downtime. 
Your sole and exclusive remedy for dissatisfaction is to stop using the application.

3. Dispute Resolution and Class Action Waiver
You agree that all disputes, claims, or controversies shall be resolved exclusively through final and binding private arbitration rather than in court. 
You expressly waive any right to bring or participate in any class action lawsuit against the company.

4. Margin Calls and Asset Liquidation
In the event your account equity falls below maintenance margin requirements, the broker reserves the right to freeze your account and liquidate your holdings without prior notice. 
Recurring platform maintenance charges will be billed automatically to your designated payment method every month and are non-refundable.`;

// State
let currentResults = [];
let cachedModelData = null;

// DOM Elements
const contractTextEl = document.getElementById("contract-text");
const btnLoadSample = document.getElementById("btn-load-sample");
const btnClear = document.getElementById("btn-clear");
const btnAnalyze = document.getElementById("btn-analyze");
const btnText = document.getElementById("btn-text");
const btnSpinner = document.getElementById("btn-spinner");

const thresholdSlider = document.getElementById("threshold-slider");
const thresholdVal = document.getElementById("threshold-val");
const uncertaintySlider = document.getElementById("uncertainty-slider");
const uncertaintyVal = document.getElementById("uncertainty-val");

const resultsSection = document.getElementById("results-section");
const metricTotal = document.getElementById("metric-total");
const metricFlagged = document.getElementById("metric-flagged");
const metricUncertain = document.getElementById("metric-uncertain");
const metricRate = document.getElementById("metric-rate");
const filterPillsContainer = document.getElementById("filter-pills");
const clausesContainer = document.getElementById("clauses-container");

// Slider listeners
thresholdSlider.addEventListener("input", (e) => {
  thresholdVal.textContent = parseFloat(e.target.value).toFixed(2);
});

uncertaintySlider.addEventListener("input", (e) => {
  uncertaintyVal.textContent = parseFloat(e.target.value).toFixed(2);
});

// Quick action buttons
btnLoadSample.addEventListener("click", () => {
  contractTextEl.value = SAMPLE_TEXT;
  contractTextEl.focus();
});

btnClear.addEventListener("click", () => {
  contractTextEl.value = "";
  resultsSection.classList.add("hidden");
  currentResults = [];
});

// Analyze Click Handler
btnAnalyze.addEventListener("click", async () => {
  const text = contractTextEl.value.trim();
  if (!text) {
    alert("Please paste contract text or click 'Load Sample T&C' first.");
    return;
  }

  const threshold = parseFloat(thresholdSlider.value);
  const uncertainty = parseFloat(uncertaintySlider.value);

  setLoading(true);

  try {
    let data = null;

    // 1. Attempt Vercel Serverless API Call
    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text,
          threshold: threshold,
          uncertainty_band: uncertainty
        })
      });

      if (response.ok) {
        data = await response.json();
      }
    } catch (apiErr) {
      console.warn("Serverless API unreachable, falling back to client-side neural network inference.", apiErr);
    }

    // 2. Client-Side Fallback if API returned null or was unreachable
    if (!data || !data.clauses) {
      data = await runClientSideInference(text, threshold, uncertainty);
    }

    renderResults(data);
  } catch (err) {
    console.error("Analysis failure:", err);
    alert("Error analyzing clauses: " + err.message);
  } finally {
    setLoading(false);
  }
});

function setLoading(isLoading) {
  if (isLoading) {
    btnText.textContent = "Analyzing...";
    btnSpinner.classList.remove("hidden");
    btnAnalyze.disabled = true;
  } else {
    btnText.textContent = "🔍 Analyze Clauses";
    btnSpinner.classList.add("hidden");
    btnAnalyze.disabled = false;
  }
}

// Client-Side Exact Mathematical Forward Pass (Zero Dependencies)
async function runClientSideInference(text, threshold, uncertainty) {
  if (!cachedModelData) {
    const res = await fetch("model_data.json");
    if (!res.ok) throw new Error("Could not load neural network weights.");
    cachedModelData = await res.json();
  }

  const clauses = segmentTextClient(text);
  const vocab = cachedModelData.vocab;
  const idf = cachedModelData.idf;
  const weights = cachedModelData.weights;

  const results = [];
  for (const c of clauses) {
    const xVec = computeTfIdfClient(c, vocab, idf);
    const prob = runAnnForwardPass(xVec, weights);

    const dist = Math.abs(prob - threshold);
    const isUncertain = dist <= uncertainty;
    let isFlagged = prob >= threshold || isUncertain;

    let status = "Standard / Not Flagged";
    if (prob >= threshold) {
      status = isUncertain ? "Uncertain: Review Manually" : "Flagged: Potentially Risky";
    } else if (isUncertain) {
      status = "Uncertain: Review Manually";
    }

    const categories = isFlagged ? tagClauseClient(c) : [];
    const primaryCat = categories[0] || "general_risk";

    const explanations = {
      unilateral_change: "it allows the service provider to modify fees, features, or contract obligations at any time without requiring your prior consent.",
      limitation_of_liability: "it restricts or disclaims provider liability, eliminates warranties, or requires you to accept services 'as-is' with limited legal recourse.",
      arbitration: "it waives your right to pursue disputes in court or participate in consumer class action lawsuits, requiring binding private arbitration.",
      unilateral_termination: "it gives the company sole discretion to terminate or suspend your account and access without prior warning or cause.",
      account_freeze_liquidation: "it permits the platform to freeze your funds or forcibly liquidate your assets/positions without an adequate grace period.",
      auto_renewal: "it enrolls you in automatic recurring billing or non-refundable renewal charges unless cancelled within narrow deadlines.",
      data_sharing: "it allows sharing or transferring your personal or financial records to third parties and affiliated companies.",
      general_risk: "its language and contractual structure statistically match terms known to create significant imbalance against the consumer."
    };

    results.append ? null : results.push({
      clause_text: c,
      predicted_probability: parseFloat(prob.toFixed(4)),
      is_flagged: isFlagged,
      is_uncertain: isUncertain,
      status_label: status,
      categories: categories,
      primary_category: primaryCat,
      explanation: `This clause may matter because ${explanations[primaryCat] || explanations.general_risk}`,
      reference_note: null
    });
  }

  const flaggedCount = results.filter(r => r.is_flagged).length;
  const uncertainCount = results.filter(r => r.is_uncertain).length;

  return {
    clauses: results,
    summary: {
      total: results.length,
      flagged: flaggedCount,
      uncertain: uncertainCount
    }
  };
}

function segmentTextClient(text) {
  const cleaned = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  const splitPattern = /(?<=[.!?])\s+(?=[A-Z0-9\(\[\"'])|\n\s*\n+|(?<=\n)\s*(?=[0-9]+[\.\)]\s+|[a-z][\.\)]\s+|\([0-9a-z]+\)\s+|[-•*]\s+)/g;
  return cleaned
    .split(splitPattern)
    .map(c => c.replace(/\s+/g, ' ').trim())
    .filter(c => c.length >= 10);
}

function computeTfIdfClient(text, vocab, idf) {
  const matches = text.toLowerCase().match(/\b\w\w+\b/g) || [];
  const ngrams = [...matches];
  for (let i = 0; i < matches.length - 1; i++) {
    ngrams.push(matches[i] + ' ' + matches[i+1]);
  }

  const counts = {};
  for (const ng of ngrams) {
    if (vocab[ng] !== undefined) {
      counts[ng] = (counts[ng] || 0) + 1;
    }
  }

  const vec = new Float32Array(5000);
  let sumSq = 0;

  for (const ng in counts) {
    const idx = vocab[ng];
    const tf = 1.0 + Math.log(counts[ng]);
    const val = tf * idf[idx];
    vec[idx] = val;
    sumSq += val * val;
  }

  if (sumSq > 0) {
    const norm = Math.sqrt(sumSq);
    for (let i = 0; i < vec.length; i++) {
      if (vec[i] !== 0) vec[i] /= norm;
    }
  }
  return vec;
}

function runAnnForwardPass(x, weights) {
  const { fc1_w, fc1_b, fc2_w, fc2_b, fc_out_w, fc_out_b } = weights;
  
  // Layer 1 (256 units) + ReLU
  const h1 = new Float32Array(256);
  for (let i = 0; i < 256; i++) {
    let dot = fc1_b[i];
    const row = fc1_w[i];
    for (let j = 0; j < 5000; j++) {
      if (x[j] !== 0) dot += row[j] * x[j];
    }
    h1[i] = dot > 0 ? dot : 0;
  }

  // Layer 2 (64 units) + ReLU
  const h2 = new Float32Array(64);
  for (let k = 0; k < 64; k++) {
    let dot = fc2_b[k];
    const row = fc2_w[k];
    for (let j = 0; j < 256; j++) {
      if (h1[j] !== 0) dot += row[j] * h1[j];
    }
    h2[k] = dot > 0 ? dot : 0;
  }

  // Output (1 logit)
  let logit = fc_out_b[0];
  const outRow = fc_out_w[0];
  for (let k = 0; k < 64; k++) {
    if (h2[k] !== 0) logit += outRow[k] * h2[k];
  }

  const clampedLogit = Math.max(-30, Math.min(30, logit));
  return 1.0 / (1.0 + Math.exp(-clampedLogit));
}

function tagClauseClient(text) {
  const patterns = {
    unilateral_change: [/modify\s+(these|the|our)?\s*terms/i, /change\s+(these|the|our)?\s*(terms|prices|fees)/i, /at\s+any\s+time\s+without\s+notice/i, /reserves?\s+the\s+right\s+to\s+(change|amend|modify)/i],
    limitation_of_liability: [/limitation\s+of\s+liability/i, /as-is/i, /as\s+available/i, /disclaim\w*\s+all\s+warranties/i, /not\s+liable\s+for/i, /hold\s+harmless/i, /indemnif/i],
    arbitration: [/mandatory\s+arbitration/i, /binding\s+arbitration/i, /waive\w*\s+the\s+right\s+to\s+a\s+jury\s+trial/i, /class\s+action\s+waiver/i],
    unilateral_termination: [/terminate\s+your\s+(access|account)/i, /suspend\s+(or\s+terminate\s+)?your\s+account/i, /in\s+our\s+sole\s+discretion/i],
    account_freeze_liquidation: [/freeze\s+(your\s+)?(account|funds|assets)/i, /liquidat\w*\s+(your\s+)?(holdings|positions|assets)/i, /margin\s+call/i],
    auto_renewal: [/automatic(ally)?\s+renew/i, /recurring\s+(charge|fee|payment)/i, /non-refundable/i],
    data_sharing: [/share\s+your\s+(personal|financial)\s+data/i, /third-party\s+partners/i]
  };

  const matched = [];
  for (const cat in patterns) {
    for (const p of patterns[cat]) {
      if (p.test(text)) {
        matched.push(cat);
        break;
      }
    }
  }
  return matched.length ? matched : ["general_risk"];
}

// Render Results to UI
function renderResults(data) {
  currentResults = data.clauses || [];
  const summary = data.summary || { total: 0, flagged: 0, uncertain: 0 };

  resultsSection.classList.remove("hidden");

  // Update Metric Cards
  metricTotal.textContent = summary.total;
  metricFlagged.textContent = summary.flagged;
  metricUncertain.textContent = summary.uncertain;
  metricRate.textContent = summary.total > 0 
    ? `${((summary.flagged / summary.total) * 100).toFixed(1)}%` 
    : "0%";

  // Build Category Filters
  const flaggedClauses = currentResults.filter(r => r.is_flagged);
  const categories = new Set();
  flaggedClauses.forEach(r => r.categories.forEach(c => categories.add(c)));

  filterPillsContainer.innerHTML = '<button class="pill active" data-cat="all">All Flagged</button>';
  categories.forEach(cat => {
    const btn = document.createElement("button");
    btn.className = "pill";
    btn.dataset.cat = cat;
    btn.textContent = cat.replace(/_/g, " ");
    filterPillsContainer.appendChild(btn);
  });

  // Attach filter pill event listeners
  filterPillsContainer.querySelectorAll(".pill").forEach(pill => {
    pill.addEventListener("click", () => {
      filterPillsContainer.querySelectorAll(".pill").forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      filterAndRenderClauses(pill.dataset.cat);
    });
  });

  // Initial render of all flagged clauses sorted descending by probability
  filterAndRenderClauses("all");

  // Smooth scroll into results
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function filterAndRenderClauses(category) {
  clausesContainer.innerHTML = "";

  const flagged = currentResults.filter(r => r.is_flagged);
  if (flagged.length === 0) {
    clausesContainer.innerHTML = `
      <div class="card" style="text-align: center; color: var(--text-muted); padding: 32px;">
        <h3>ℹ️ No clauses met the flagging criteria at this threshold.</h3>
        <p style="margin-top: 8px;">Reminder: This does <strong>NOT</strong> certify that the document is safe. You can lower the sensitivity threshold slider to inspect borderline clauses.</p>
      </div>
    `;
    return;
  }

  // Sort descending by probability
  flagged.sort((a, b) => b.predicted_probability - a.predicted_probability);

  const filtered = category === "all"
    ? flagged
    : flagged.filter(r => r.categories.includes(category));

  filtered.forEach((item, idx) => {
    const card = document.createElement("div");
    const isHighRisk = !item.is_uncertain;
    card.className = `clause-card ${isHighRisk ? 'flagged' : 'uncertain'} ${idx < 3 ? 'expanded' : ''}`;

    const probPct = (item.predicted_probability * 100).toFixed(1);
    const badgeClass = isHighRisk ? 'danger' : 'warning';
    const tagsHtml = item.categories.map(c => `<span class="cat-tag">${c.replace(/_/g, ' ')}</span>`).join('');

    card.innerHTML = `
      <div class="clause-card-header">
        <div class="clause-header-left">
          <span class="score-badge ${badgeClass}">${probPct}% Risk</span>
          <span class="clause-preview">${escapeHtml(item.clause_text)}</span>
        </div>
        <span class="chevron-icon">▼</span>
      </div>
      <div class="clause-card-body">
        <div class="tags-row">${tagsHtml}</div>
        <div class="clause-full-text">"${escapeHtml(item.clause_text)}"</div>
        <div class="explanation-box">
          <strong>Impact Explanation:</strong> ${escapeHtml(item.explanation)}
        </div>
      </div>
    `;

    // Toggle expand/collapse
    card.querySelector(".clause-card-header").addEventListener("click", () => {
      card.classList.toggle("expanded");
    });

    clausesContainer.appendChild(card);
  });
}

function escapeHtml(str) {
  if (!str) return "";
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
