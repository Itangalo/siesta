const SUIT_GLYPHS = {
  hearts: "♥",
  diamonds: "♦",
  clubs: "♣",
  spades: "♠",
};

const SUIT_LABELS = {
  hearts: "HJ",
  diamonds: "RU",
  clubs: "KL",
  spades: "SP",
};

const SUIT_CODES = {
  hearts: "H",
  diamonds: "D",
  clubs: "C",
  spades: "S",
};

const SUIT_ROW_ORDER = [
  { suit: "spades", label: "Spader" },
  { suit: "hearts", label: "Hjarter" },
  { suit: "clubs", label: "Klover" },
  { suit: "diamonds", label: "Ruter" },
];

const RANK_ORDER = [13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];

const RANK_LABELS = {
  1: "A",
  11: "J",
  12: "Q",
  13: "K",
};

const FEEDBACK_LABELS = {
  normal: "sparad",
  important: "viktig korrigering",
  key_move: "nyckeldrag",
};

const ACTIVE_GAME_STORAGE_KEY = "siesta.activeGameId";
const CAPTURE_STRENGTH_STORAGE_KEY = "siesta.captureStrength";
const MARK_DEAD_STORAGE_KEY = "siesta.markDead";
const MARK_MOVABLE_STORAGE_KEY = "siesta.markMovable";
const MARK_REACHABLE_STORAGE_KEY = "siesta.markReachable";
const REACHABLE_DEPTH = 4;
const AI_REQUEST_TIMEOUT_MS = 3500;

const state = {
  gameId: null,
  snapshot: null,
  selection: null,
  selectionTarget: null,
  suggestions: [],
  aiContext: null,
  feedbackStatus: "",
  savedFeedbackKeys: new Set(),
  lastFeedback: null,
  hoveredCardCode: null,
  hoveredSuggestion: null,
  aiRequestId: 0,
  aiLoading: false,
  aiAbortController: null,
  lastFeedbackStrength: null,
  captureStrength: "normal",
  markDead: false,
  markMovable: false,
  markReachable: false,
  reachable: null,
  reachableStateHash: null,
};

/**
 * Fetch the multi-step reachability marking for the current position.
 *
 * Kept off the snapshot on purpose: at depth 4 this costs up to ~340ms on a
 * branchy position, which would be paid on every move. The result is tagged
 * with the state hash it was computed for, so a late reply for a position we
 * have already left is discarded rather than drawn.
 */
async function refreshReachable() {
  if (!state.markReachable || !state.gameId || !state.snapshot) {
    state.reachable = null;
    state.reachableStateHash = null;
    return;
  }
  const requestedHash = state.snapshot.state_hash;
  try {
    const result = await api(
      `/game/reachable-moves?game_id=${encodeURIComponent(state.gameId)}&depth=${REACHABLE_DEPTH}`
    );
    if (!state.snapshot || state.snapshot.state_hash !== requestedHash) {
      return;
    }
    state.reachable = result.movable_within;
    state.reachableStateHash = result.state_hash;
    setReachableError(null);
    renderBoard();
  } catch (error) {
    state.reachable = null;
    state.reachableStateHash = null;
    // A silent failure is indistinguishable from "nothing is reachable", which
    // is the worst possible way for this to break.
    setReachableError(error);
    renderBoard();
  }
}

function setReachableError(error) {
  const label = document.getElementById("mark-reachable").closest(".toggle");
  if (!error) {
    label.classList.remove("failed");
    label.removeAttribute("title");
    return;
  }
  label.classList.add("failed");
  label.title = "Kunde inte hämta flerstegsmarkeringen. Är servern omstartad efter senaste ändringen?";
  console.warn("[siesta] reachable-moves failed:", error);
}

function readStoredFlag(key) {
  try {
    return window.localStorage.getItem(key) === "true";
  } catch (_) {
    return false;
  }
}

function storeFlag(key, value) {
  try {
    window.localStorage.setItem(key, value ? "true" : "false");
  } catch (_) {
    // Ignore storage errors; the game still works without persistence.
  }
}

/**
 * Cards that have an actual legal destination right now, as "column:index".
 *
 * Deliberately not the same as isMovableCard(), which only says a card sits on
 * top of a colour run and can be picked up. Roughly four in ten pickable cards
 * have nowhere to go.
 */
function movableCardKeys(snapshot) {
  const keys = new Set();
  (snapshot.legal_moves || []).forEach((move) => {
    if (move.type !== "move") {
      return;
    }
    const height = snapshot.columns[move.from_column].cards.length;
    for (let index = height - move.run_length; index < height; index += 1) {
      keys.add(`${move.from_column}:${index}`);
    }
  });
  return keys;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Request failed");
  }
  return data;
}

function suitGlyph(card) {
  return SUIT_GLYPHS[card.suit] || card.suit_symbol;
}

function suitLabel(card) {
  return SUIT_LABELS[card.suit] || card.suit_symbol;
}

function cardSymbol(card) {
  return `${card.rank_label}${suitGlyph(card)}`;
}

function rankLabel(rank) {
  return RANK_LABELS[rank] || String(rank);
}

function cardCode(rank, suit) {
  return `${rankLabel(rank)}${SUIT_CODES[suit]}`;
}

function cardMarkup(card) {
  return `
    <span class="card-corner suit-${card.suit}" aria-hidden="true">
      <span class="card-rank">${card.rank_label}</span>
      <span class="card-suit">${suitGlyph(card)}</span>
      <span class="card-suit-label">${suitLabel(card)}</span>
    </span>
    <span class="card-watermark suit-${card.suit}" aria-hidden="true">${suitGlyph(card)}</span>
    <span class="card-face" aria-hidden="true">${cardSymbol(card)}</span>
  `;
}

function autoAiEnabled() {
  return document.getElementById("auto-ai").checked;
}

function readStoredGameId() {
  try {
    return window.localStorage.getItem(ACTIVE_GAME_STORAGE_KEY);
  } catch (_) {
    return null;
  }
}

function readUrlGameId() {
  try {
    const url = new URL(window.location.href);
    return url.searchParams.get("game_id");
  } catch (_) {
    return null;
  }
}

function storeActiveGameId(gameId) {
  try {
    if (gameId) {
      window.localStorage.setItem(ACTIVE_GAME_STORAGE_KEY, gameId);
    } else {
      window.localStorage.removeItem(ACTIVE_GAME_STORAGE_KEY);
    }
  } catch (_) {
    // Ignore storage errors; the game still works without persistence.
  }
}

function readStoredCaptureStrength() {
  try {
    const value = window.localStorage.getItem(CAPTURE_STRENGTH_STORAGE_KEY);
    return value === "important" || value === "key_move" ? value : "normal";
  } catch (_) {
    return "normal";
  }
}

function storeCaptureStrength(strength) {
  try {
    window.localStorage.setItem(CAPTURE_STRENGTH_STORAGE_KEY, strength);
  } catch (_) {
    // Ignore storage errors; the game still works without persistence.
  }
}

function syncUrlGameId(gameId) {
  try {
    const url = new URL(window.location.href);
    if (gameId) {
      url.searchParams.set("game_id", gameId);
    } else {
      url.searchParams.delete("game_id");
    }
    window.history.replaceState({}, "", url);
  } catch (_) {
    // Ignore URL update failures.
  }
}

function cancelPendingAiRequest() {
  if (state.aiAbortController) {
    state.aiAbortController.abort();
    state.aiAbortController = null;
  }
  state.aiLoading = false;
}

function actionKey(action) {
  if (!action) {
    return "";
  }
  if (action.type === "deal") {
    return "deal";
  }
  return `${action.type}:${action.from_column}:${action.to_column}:${action.run_length}`;
}

function feedbackKey(stateHash, action, strength, appliesTo) {
  return `${stateHash}:${actionKey(action)}:${strength}:${appliesTo}`;
}

function rankingSource() {
  if (!state.aiContext) {
    return "search";
  }
  return state.aiContext.rankingSource || (state.aiContext.modelEligible ? "model" : "search");
}

function selectedRunLength(columnIndex, cardIndex) {
  const column = state.snapshot.columns[columnIndex];
  return column.cards.length - cardIndex;
}

function isMovableCard(column, cardIndex) {
  return column.cards.length - cardIndex <= column.movable_run_length;
}

function chosenMoveAction(fromColumn, toColumn, runLength) {
  return {
    type: "move",
    from_column: fromColumn,
    to_column: toColumn,
    run_length: runLength,
    description: `kol ${fromColumn + 1} -> kol ${toColumn + 1} (${runLength})`,
  };
}

function dealAction() {
  return { type: "deal", description: `Deal from stock (${state.snapshot.stock_count} left)` };
}

function buildFeedbackEntry(chosenAction) {
  const topSuggestion = state.aiContext && state.aiContext.suggestions.length ? state.aiContext.suggestions[0] : null;
  return {
    gameId: state.gameId,
    stateHash: state.snapshot.state_hash,
    stateSnapshot: state.snapshot,
    aiSource: state.aiContext ? rankingSource() : "none",
    modelLoaded: state.aiContext ? state.aiContext.modelLoaded : false,
    modelEligible: state.aiContext ? state.aiContext.modelEligible : false,
    topSuggestion,
    chosenAction,
    candidateActions: state.aiContext ? state.aiContext.suggestions : [],
  };
}

function currentPlannedFeedbackEntry() {
  if (!state.selection || state.selectionTarget === null) {
    return null;
  }
  const chosenAction = chosenMoveAction(state.selection.fromColumn, state.selectionTarget, state.selection.runLength);
  return buildFeedbackEntry(chosenAction);
}

function updateFeedbackPanel() {
  const summary = document.getElementById("ai-summary");
  const pendingPanel = document.getElementById("pending-feedback-panel");
  const pendingText = document.getElementById("pending-feedback-text");
  const lastPanel = document.getElementById("board-callout");
  const lastText = document.getElementById("last-feedback-text");
  const status = document.getElementById("feedback-status");
  status.textContent = state.feedbackStatus;

  if (state.aiLoading && (!state.aiContext || state.aiContext.stateHash !== state.snapshot?.state_hash)) {
    summary.textContent = "AI analyserar nuvarande läge...";
    pendingPanel.classList.add("hidden");
    lastPanel.classList.add("hidden");
    pendingText.textContent = "";
    lastText.textContent = "";
    return;
  }

  if (!state.aiContext || !state.aiContext.suggestions.length) {
    summary.textContent = autoAiEnabled()
      ? "AI auto är på. Förslag laddas automatiskt för nuvarande läge."
      : "Ladda AI-förslag för att se hur agenten resonerar.";
    pendingPanel.classList.add("hidden");
    pendingText.textContent = "";
  } else {
    const sourceMap = {
      model: "modellen",
      search: "search-läraren",
      "model+hole": "modellen + hålmodellen",
      "search+hole": "search + hålmodellen",
      "model+compact": "modellen + compactness-modellen",
      "search+compact": "search + compactness-modellen",
      "model+hole+compact": "modellen + hålmodellen + compactness-modellen",
      "search+hole+compact": "search + hålmodellen + compactness-modellen",
    };
    const source = sourceMap[rankingSource()] || rankingSource();
    const top = state.aiContext.suggestions[0];
    const searchPart = top.search_score === null || top.search_score === undefined ? "" : ` | search ${top.search_score.toFixed(1)}`;
    const modelPart = top.model_score === null || top.model_score === undefined ? "" : ` | modell ${top.model_score.toFixed(2)}`;
    const holePart = top.hole_model_score === null || top.hole_model_score === undefined ? "" : ` | hål ${top.hole_model_score.toFixed(2)}`;
    const compactPart = top.compact_model_score === null || top.compact_model_score === undefined ? "" : ` | kompakt ${top.compact_model_score.toFixed(2)}`;
    summary.textContent = `Aktiv AI-källa: ${source}. Toppförslag: ${top.description}${searchPart}${modelPart}${holePart}${compactPart}.`;

    const candidate = currentPlannedFeedbackEntry();
    if (!candidate || !candidate.topSuggestion || actionKey(candidate.chosenAction) === actionKey(candidate.topSuggestion)) {
      pendingPanel.classList.add("hidden");
      pendingText.textContent = "";
    } else {
      pendingPanel.classList.remove("hidden");
      pendingText.textContent = `Om du tänker spela ${candidate.chosenAction.description} i stället för AI:ns toppförslag ${candidate.topSuggestion.description}, markera det här innan du gör draget.`;
    }
  }

  if (!state.lastFeedback) {
    lastPanel.classList.add("hidden");
    lastText.textContent = "";
    updateLastFeedbackButtons();
    return;
  }

  lastPanel.classList.remove("hidden");
  lastText.textContent = `${state.lastFeedback.chosenAction.description}. Draget är redan sparat som träningsdata.`;
  updateLastFeedbackButtons();
}

function updateLastFeedbackButtons() {
  const strength = state.lastFeedbackStrength || "normal";
  const mapping = {
    normal: "last-normal",
    important: "last-important",
    key_move: "last-key",
  };
  Object.values(mapping).forEach((id) => {
    document.getElementById(id).classList.remove("active");
    document.getElementById(id).setAttribute("aria-pressed", "false");
  });
  const activeId = mapping[strength];
  if (activeId) {
    document.getElementById(activeId).classList.add("active");
    document.getElementById(activeId).setAttribute("aria-pressed", "true");
  }
}

function updateCaptureButtons() {
  const mapping = {
    normal: "capture-normal",
    important: "capture-important",
    key_move: "capture-key",
  };
  Object.values(mapping).forEach((id) => {
    document.getElementById(id).classList.remove("active");
    document.getElementById(id).setAttribute("aria-pressed", "false");
  });
  const activeId = mapping[state.captureStrength || "normal"];
  if (activeId) {
    document.getElementById(activeId).classList.add("active");
    document.getElementById(activeId).setAttribute("aria-pressed", "true");
  }
}

function setCaptureStrength(strength) {
  state.captureStrength = strength;
  storeCaptureStrength(strength);
  updateCaptureButtons();
}

function updateLostBanner() {
  const banner = document.getElementById("lost-banner");
  const locked = state.snapshot && state.snapshot.provably_lost && state.snapshot.status === "in_progress";
  banner.classList.toggle("hidden", !locked);
  if (locked) {
    banner.textContent =
      "Partiet är kört. Varje kolumn har ett dött kort, så inget hål kan skapas igen – ge upp och blanda om.";
  }
}

function updateStatus() {
  const snapshot = state.snapshot;
  document.getElementById("status-text").textContent = snapshot.status;
  updateLostBanner();
  document.getElementById("stock-count").textContent = snapshot.stock_count;
  document.getElementById("sequence-count").textContent = snapshot.completed_sequences;
  document.getElementById("move-count").textContent = snapshot.moves_played;
  const selection = state.selection;
  document.getElementById("selection-text").textContent = selection
    ? `Vald sekvens: kolumn ${selection.fromColumn + 1}, längd ${selection.runLength}. Klicka eller släpp på målkolumn.`
    : "Ingen flytt vald.";
  updateFeedbackPanel();
  updateCaptureButtons();
}

function renderSuggestions() {
  const container = document.getElementById("suggestions");
  if (!state.suggestions.length) {
    container.className = "suggestions empty";
    container.textContent = state.aiLoading ? "AI analyserar..." : "Inga AI-förslag laddade.";
    return;
  }
  container.className = "suggestions";
  container.innerHTML = state.suggestions
    .map((suggestion) => {
      let sourceBadge = state.aiContext && state.aiContext.modelEligible ? "modell" : "search";
      if (state.aiContext && state.aiContext.holeModelActive) {
        sourceBadge += "+hål";
      }
      if (state.aiContext && state.aiContext.compactModelActive) {
        sourceBadge += "+kompakt";
      }
      const searchPart =
        suggestion.search_score === null || suggestion.search_score === undefined
          ? ""
          : ` | search ${suggestion.search_score.toFixed(1)}`;
      const modelPart =
        suggestion.model_score === null || suggestion.model_score === undefined
          ? ""
          : ` | modell ${suggestion.model_score.toFixed(3)}`;
      const holePart =
        suggestion.hole_model_score === null || suggestion.hole_model_score === undefined
          ? ""
          : ` | hål ${suggestion.hole_model_score.toFixed(3)}`;
      const compactPart =
        suggestion.compact_model_score === null || suggestion.compact_model_score === undefined
          ? ""
          : ` | kompakt ${suggestion.compact_model_score.toFixed(3)}`;
      const lockPart = suggestion.locks_position ? ` | <span class="locks-position">låser partiet</span>` : "";
      return `<button class="suggestion-item${suggestion.locks_position ? " locks" : ""}" data-type="${suggestion.type}" data-from="${suggestion.from_column ?? ""}" data-to="${suggestion.to_column ?? ""}" data-run="${suggestion.run_length ?? ""}">
        <span class="suggestion-main">${suggestion.description}</span>
        <span class="suggestion-meta">${sourceBadge} | heuristik ${suggestion.heuristic_score.toFixed(1)}${searchPart}${modelPart}${holePart}${compactPart}${lockPart}</span>
      </button>`;
    })
    .join("");

  container.querySelectorAll(".suggestion-item").forEach((button) => {
    button.addEventListener("mouseenter", () => {
      const type = button.dataset.type;
      state.hoveredSuggestion = {
        type,
        fromColumn: type === "move" ? Number(button.dataset.from) : null,
        toColumn: type === "move" ? Number(button.dataset.to) : null,
        runLength: type === "move" ? Number(button.dataset.run) : null,
      };
      renderBoard();
    });
    button.addEventListener("mouseleave", () => {
      state.hoveredSuggestion = null;
      renderBoard();
    });
    button.addEventListener("click", async () => {
      if (button.dataset.type === "deal") {
        await performDeal();
        return;
      }
      state.selectionTarget = Number(button.dataset.to);
      await performMove(Number(button.dataset.from), Number(button.dataset.to), Number(button.dataset.run));
    });
  });
}

function syncHoveredCardHighlight() {
  document.querySelectorAll("[data-card-code]").forEach((element) => {
    const matches = !!state.hoveredCardCode && element.dataset.cardCode === state.hoveredCardCode;
    element.classList.toggle("hover-match", matches);
  });
}

function renderStockOverview() {
  const container = document.getElementById("stock-overview");
  if (!state.snapshot) {
    container.innerHTML = "";
    return;
  }

  const dealtCards = new Set();
  state.snapshot.columns.forEach((column) => {
    column.cards.forEach((card) => {
      dealtCards.add(card.code);
    });
  });

  container.innerHTML = SUIT_ROW_ORDER.map(({ suit, label }) => {
    const cells = RANK_ORDER.map((rank) => {
      const code = cardCode(rank, suit);
      const inStock = !dealtCards.has(code);
      return `<span class="stock-card suit-${suit} ${inStock ? "in-stock" : "dealt"}" data-card-code="${code}" title="${code}">
        <span class="stock-rank">${rankLabel(rank)}</span>
        <span class="stock-suit">${SUIT_GLYPHS[suit]}</span>
      </span>`;
    }).join("");

    return `<div class="stock-row">
      <div class="stock-row-label suit-${suit}">${label} ${SUIT_GLYPHS[suit]}</div>
      <div class="stock-row-cards">${cells}</div>
    </div>`;
  }).join("");

  container.querySelectorAll(".stock-card").forEach((cardElement) => {
    cardElement.addEventListener("mouseenter", () => {
      state.hoveredCardCode = cardElement.dataset.cardCode || null;
      syncHoveredCardHighlight();
    });
    cardElement.addEventListener("mouseleave", () => {
      state.hoveredCardCode = null;
      syncHoveredCardHighlight();
    });
  });

  syncHoveredCardHighlight();
}

function renderBoard() {
  const board = document.getElementById("board");
  board.innerHTML = "";
  const movableKeys = state.markMovable ? movableCardKeys(state.snapshot) : new Set();
  // Only trust the reachability result if it was computed for this position.
  const reachable =
    state.markReachable && state.reachableStateHash === state.snapshot.state_hash ? state.reachable : null;

  state.snapshot.columns.forEach((column, columnIndex) => {
    const columnElement = document.createElement("section");
    columnElement.className = "column";
    columnElement.dataset.columnIndex = String(columnIndex);
    columnElement.innerHTML = `<div class="column-title">Kolumn ${columnIndex + 1}</div>`;

    const stack = document.createElement("div");
    stack.className = `stack ${column.cards.length === 0 ? "empty" : ""}`;
    const suggestionTargetsColumn =
      state.hoveredSuggestion &&
      state.hoveredSuggestion.type === "move" &&
      state.hoveredSuggestion.toColumn === columnIndex;
    if (suggestionTargetsColumn) {
      stack.classList.add("suggestion-target");
    }
    const stackHeight = column.cards.length ? Math.max(416, 88 + (column.cards.length - 1) * 28 + 116) : 416;
    stack.style.height = `${stackHeight}px`;
    stack.addEventListener("dragover", (event) => event.preventDefault());
    stack.addEventListener("drop", async (event) => {
      event.preventDefault();
      const fromColumn = Number(event.dataTransfer.getData("fromColumn"));
      const runLength = Number(event.dataTransfer.getData("runLength"));
      if (Number.isFinite(fromColumn) && Number.isFinite(runLength)) {
        state.selectionTarget = columnIndex;
        await performMove(fromColumn, columnIndex, runLength);
      }
    });
    stack.addEventListener("click", async () => {
      if (state.selection) {
        state.selectionTarget = columnIndex;
        updateFeedbackPanel();
        await performMove(state.selection.fromColumn, columnIndex, state.selection.runLength);
      }
    });

    if (!column.cards.length) {
      const hole = document.createElement("div");
      hole.className = "hole";
      hole.textContent = "Hål";
      stack.appendChild(hole);
    } else {
      column.cards.forEach((card, cardIndex) => {
        const cardElement = document.createElement("div");
        const movable = isMovableCard(column, cardIndex);
        const runLength = movable ? selectedRunLength(columnIndex, cardIndex) : 0;
        const selected =
          state.selection &&
          state.selection.fromColumn === columnIndex &&
          state.selection.runLength === runLength &&
          movable;
        const isSuggestedSource =
          state.hoveredSuggestion &&
          state.hoveredSuggestion.type === "move" &&
          state.hoveredSuggestion.fromColumn === columnIndex &&
          column.cards.length - cardIndex <= state.hoveredSuggestion.runLength;
        const isSuggestedDestination =
          state.hoveredSuggestion &&
          state.hoveredSuggestion.type === "move" &&
          state.hoveredSuggestion.toColumn === columnIndex &&
          cardIndex === column.cards.length - 1;

        cardElement.className = `card ${card.color} suit-${card.suit} ${movable ? "movable" : ""} ${selected ? "selected" : ""}`;
        if (state.markMovable && movableKeys.has(`${columnIndex}:${cardIndex}`)) {
          cardElement.classList.add("has-move");
        }
        // Depth 1 is already covered by .has-move, so only mark what needs
        // more than one move - otherwise the two markings say the same thing.
        const reachDepth = reachable ? reachable[card.code] : undefined;
        if (reachDepth && reachDepth > 1) {
          cardElement.classList.add("reachable-move");
          cardElement.dataset.reachDepth = String(reachDepth);
          cardElement.title = `Kan flyttas efter ${reachDepth} drag.`;
        }
        if (state.markDead && card.dead_marked) {
          cardElement.classList.add("dead-card");
          cardElement.title = card.dead_by_rule
            ? "Dött kort: kan bara flyttas till ett hål."
            : "Kung: död per definition, kan bara flyttas till ett hål.";
        }
        if (isSuggestedSource) {
          cardElement.classList.add("suggestion-source");
        }
        if (isSuggestedDestination) {
          cardElement.classList.add("suggestion-destination");
        }
        cardElement.dataset.cardCode = card.code;
        cardElement.innerHTML = cardMarkup(card);
        if (state.markDead && card.dead_marked) {
          // Must come after innerHTML, which would otherwise wipe the badge.
          const badge = document.createElement("span");
          badge.className = `dead-badge ${card.dead_by_rule ? "" : "by-definition"}`;
          badge.textContent = "✝";
          badge.setAttribute("aria-hidden", "true");
          cardElement.appendChild(badge);
        }
        cardElement.style.top = `${cardIndex * 28}px`;
        if (movable) {
          cardElement.draggable = true;
          cardElement.addEventListener("dragstart", (event) => {
            event.dataTransfer.setData("fromColumn", String(columnIndex));
            event.dataTransfer.setData("runLength", String(runLength));
          });
          cardElement.addEventListener("click", (event) => {
            event.stopPropagation();
            if (selected) {
              state.selection = null;
              state.selectionTarget = null;
            } else {
              state.selection = { fromColumn: columnIndex, runLength };
            }
            updateStatus();
            renderBoard();
          });
        }
        stack.appendChild(cardElement);
      });
    }

    columnElement.appendChild(stack);
    board.appendChild(columnElement);
  });

  syncHoveredCardHighlight();
}

function render() {
  updateStatus();
  renderBoard();
  renderStockOverview();
  renderSuggestions();
}

function applySnapshot(snapshot) {
  state.snapshot = snapshot;
  state.gameId = snapshot.game_id;
  state.selection = null;
  state.selectionTarget = null;
  state.hoveredSuggestion = null;
  storeActiveGameId(snapshot.game_id);
  syncUrlGameId(snapshot.game_id);
  render();
  refreshReachable();
  if (autoAiEnabled() && snapshot.status === "in_progress") {
    loadSuggestions({ silentIfCurrent: true }).catch((error) => {
      state.feedbackStatus = error.message;
      updateFeedbackPanel();
    });
  }
}

async function startNewGame() {
  cancelPendingAiRequest();
  const snapshot = await api("/game/new", {
    method: "POST",
    body: JSON.stringify({}),
  });
  state.suggestions = [];
  state.aiContext = null;
  state.feedbackStatus = "";
  state.savedFeedbackKeys = new Set();
  state.lastFeedback = null;
  state.lastFeedbackStrength = null;
  state.captureStrength = readStoredCaptureStrength();
  applySnapshot(snapshot);
}

async function restoreStoredGame() {
  const gameId = readUrlGameId() || readStoredGameId();
  if (!gameId) {
    return false;
  }

  try {
    const snapshot = await api(`/game/state?game_id=${encodeURIComponent(gameId)}`);
    state.suggestions = [];
    state.aiContext = null;
    state.feedbackStatus = "Tidigare parti ateranslutet.";
    state.lastFeedback = null;
    state.lastFeedbackStrength = null;
    state.captureStrength = readStoredCaptureStrength();
    applySnapshot(snapshot);
    return true;
  } catch (error) {
    storeActiveGameId(null);
    syncUrlGameId(null);
    if (!String(error.message).includes("Unknown game_id")) {
      throw error;
    }
    return false;
  }
}

async function saveCorrection(entry, { strength = "normal", appliesTo = "last_move", note = "human correction" } = {}) {
  const key = feedbackKey(entry.stateHash, entry.chosenAction, strength, appliesTo);
  if (state.savedFeedbackKeys.has(key)) {
    return;
  }
  const payload = await api("/ai/feedback", {
    method: "POST",
    body: JSON.stringify({
      game_id: entry.gameId,
      state_hash: entry.stateHash,
      ai_source: entry.aiSource,
      model_loaded: entry.modelLoaded,
      model_eligible: entry.modelEligible,
      feedback_strength: strength,
      applies_to: appliesTo,
      recommended_action: entry.topSuggestion,
      chosen_action: entry.chosenAction,
      candidate_actions: entry.candidateActions,
      state_snapshot: entry.stateSnapshot,
      note,
    }),
  });
  state.savedFeedbackKeys.add(key);
  state.feedbackStatus = `${FEEDBACK_LABELS[strength] || strength} sparad i ${payload.path}`;
}

async function saveCurrentPlannedFeedback(strength) {
  const candidate = currentPlannedFeedbackEntry();
  if (!candidate) {
    return;
  }
  await saveCorrection(candidate, {
    strength,
    appliesTo: "planned_move",
    note: "planned human correction",
  });
  updateFeedbackPanel();
}

async function saveLastMoveFeedback(strength) {
  if (!state.lastFeedback) {
    return;
  }
  await saveCorrection(state.lastFeedback, {
    strength,
    appliesTo: "last_move",
    note: "post-move human correction",
  });
  state.lastFeedbackStrength = strength;
  updateFeedbackPanel();
}

async function performMove(fromColumn, toColumn, runLength) {
  if (state.snapshot.status !== "in_progress") {
    return;
  }
  const latestFeedback = buildFeedbackEntry(chosenMoveAction(fromColumn, toColumn, runLength));
  const snapshot = await api("/game/move", {
    method: "POST",
    body: JSON.stringify({
      game_id: state.gameId,
      from_column: fromColumn,
      to_column: toColumn,
      run_length: runLength,
    }),
  });
  state.suggestions = [];
  state.aiContext = null;
  state.lastFeedback = latestFeedback;
  state.lastFeedbackStrength = state.captureStrength;
  applySnapshot(snapshot);
  saveCorrection(latestFeedback, {
    strength: state.captureStrength,
    appliesTo: "last_move",
    note: "auto-saved human move",
  }).catch((error) => {
    state.feedbackStatus = error.message;
    updateFeedbackPanel();
  });
}

async function performDeal() {
  const latestFeedback = buildFeedbackEntry(dealAction());
  const snapshot = await api("/game/deal", {
    method: "POST",
    body: JSON.stringify({ game_id: state.gameId }),
  });
  state.suggestions = [];
  state.aiContext = null;
  state.lastFeedback = latestFeedback;
  state.lastFeedbackStrength = state.captureStrength;
  applySnapshot(snapshot);
  saveCorrection(latestFeedback, {
    strength: state.captureStrength,
    appliesTo: "last_move",
    note: "auto-saved human deal",
  }).catch((error) => {
    state.feedbackStatus = error.message;
    updateFeedbackPanel();
  });
}

async function performUndo() {
  const snapshot = await api("/game/undo", {
    method: "POST",
    body: JSON.stringify({ game_id: state.gameId }),
  });
  state.suggestions = [];
  state.aiContext = null;
  state.lastFeedback = null;
  state.lastFeedbackStrength = null;
  applySnapshot(snapshot);
}

async function performConcede() {
  const snapshot = await api("/game/concede", {
    method: "POST",
    body: JSON.stringify({ game_id: state.gameId }),
  });
  state.suggestions = [];
  state.aiContext = null;
  state.lastFeedback = null;
  state.lastFeedbackStrength = null;
  applySnapshot(snapshot);
}

async function loadSuggestions({ silentIfCurrent = false } = {}) {
  if (!state.snapshot || state.snapshot.status !== "in_progress") {
    return;
  }
  if (silentIfCurrent && state.aiContext && state.aiContext.stateHash === state.snapshot.state_hash) {
    return;
  }

  cancelPendingAiRequest();
  const requestId = ++state.aiRequestId;
  const targetStateHash = state.snapshot.state_hash;
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), AI_REQUEST_TIMEOUT_MS);
  state.aiAbortController = controller;
  state.aiLoading = true;
  renderSuggestions();
  updateFeedbackPanel();

  try {
    const payload = await api("/ai/evaluate-move", {
      method: "POST",
      body: JSON.stringify({ game_id: state.gameId }),
      signal: controller.signal,
    });
    if (requestId !== state.aiRequestId || !state.snapshot || state.snapshot.state_hash !== targetStateHash) {
      return;
    }
    state.aiLoading = false;
    state.aiAbortController = null;
    state.suggestions = payload.suggestions;
    state.aiContext = {
      stateHash: state.snapshot.state_hash,
      suggestions: payload.suggestions,
      modelLoaded: payload.model_loaded,
      modelEligible: payload.model_eligible,
      holeModelLoaded: payload.hole_model_loaded,
      holeModelActive: payload.hole_model_active,
      compactModelLoaded: payload.compact_model_loaded,
      compactModelActive: payload.compact_model_active,
      rankingSource: payload.ranking_source,
    };
    state.feedbackStatus = "";
    render();
  } catch (error) {
    if (requestId === state.aiRequestId) {
      state.aiLoading = false;
      state.aiAbortController = null;
      if (error.name === "AbortError") {
        state.feedbackStatus = "AI-analysen tog for lang tid. Du kan fortsatta spela eller prova igen.";
        renderSuggestions();
      }
      updateFeedbackPanel();
    }
    if (error.name === "AbortError") {
      return;
    }
    throw error;
  } finally {
    window.clearTimeout(timeoutId);
  }
}

function handleAutoAiToggle() {
  state.feedbackStatus = "";
  if (!autoAiEnabled()) {
    cancelPendingAiRequest();
    renderSuggestions();
  }
  if (autoAiEnabled() && state.snapshot && state.snapshot.status === "in_progress") {
    loadSuggestions({ silentIfCurrent: true }).catch((error) => {
      state.feedbackStatus = error.message;
      updateFeedbackPanel();
    });
  }
  updateFeedbackPanel();
}

document.getElementById("new-game").addEventListener("click", startNewGame);
document.getElementById("deal").addEventListener("click", performDeal);
document.getElementById("undo").addEventListener("click", performUndo);
document.getElementById("suggest").addEventListener("click", () => loadSuggestions());
document.getElementById("concede").addEventListener("click", performConcede);
document.getElementById("auto-ai").addEventListener("change", handleAutoAiToggle);

function bindMarkingToggle(elementId, storageKey, stateKey, onChange) {
  const input = document.getElementById(elementId);
  state[stateKey] = readStoredFlag(storageKey);
  input.checked = state[stateKey];
  input.addEventListener("change", () => {
    state[stateKey] = input.checked;
    storeFlag(storageKey, input.checked);
    if (state.snapshot) {
      renderBoard();
    }
    if (onChange) {
      onChange();
    }
  });
}

bindMarkingToggle("mark-dead", MARK_DEAD_STORAGE_KEY, "markDead");
bindMarkingToggle("mark-movable", MARK_MOVABLE_STORAGE_KEY, "markMovable");
bindMarkingToggle("mark-reachable", MARK_REACHABLE_STORAGE_KEY, "markReachable", refreshReachable);
document.getElementById("pending-normal").addEventListener("click", () => saveCurrentPlannedFeedback("normal"));
document.getElementById("pending-important").addEventListener("click", () => saveCurrentPlannedFeedback("important"));
document.getElementById("pending-key").addEventListener("click", () => saveCurrentPlannedFeedback("key_move"));
document.getElementById("last-normal").addEventListener("click", () => saveLastMoveFeedback("normal"));
document.getElementById("last-important").addEventListener("click", () => saveLastMoveFeedback("important"));
document.getElementById("last-key").addEventListener("click", () => saveLastMoveFeedback("key_move"));
document.getElementById("capture-normal").addEventListener("click", () => setCaptureStrength("normal"));
document.getElementById("capture-important").addEventListener("click", () => setCaptureStrength("important"));
document.getElementById("capture-key").addEventListener("click", () => setCaptureStrength("key_move"));

document.getElementById("status-text").textContent = "Laddar parti...";
state.captureStrength = readStoredCaptureStrength();
updateCaptureButtons();

restoreStoredGame().then((restored) => {
  if (restored) {
    return;
  }
  return startNewGame();
}).catch((error) => {
  document.getElementById("status-text").textContent = error.message;
});
