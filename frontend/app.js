const state = {
  gameId: null,
  snapshot: null,
  selection: null,
  selectionTarget: null,
  suggestions: [],
  aiContext: null,
  feedbackStatus: "",
  savedFeedbackKey: null,
};

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

function cardSymbol(card) {
  return `${card.rank_label}${card.suit_symbol}`;
}

function cardMarkup(card) {
  return `
    <span class="card-corner" aria-hidden="true">
      <span class="card-rank">${card.rank_label}</span>
      <span class="card-suit">${card.suit_symbol}</span>
    </span>
    <span class="card-face" aria-hidden="true">${cardSymbol(card)}</span>
  `;
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

function rankingSource() {
  if (!state.aiContext) {
    return "search";
  }
  return state.aiContext.modelEligible ? "model" : "search";
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

function currentCorrectionCandidate() {
  if (!state.aiContext || !state.aiContext.suggestions.length) {
    return null;
  }
  if (!state.selection || state.selectionTarget === null) {
    return null;
  }
  const chosenAction = chosenMoveAction(state.selection.fromColumn, state.selectionTarget, state.selection.runLength);
  const topSuggestion = state.aiContext.suggestions[0];
  if (actionKey(chosenAction) === actionKey(topSuggestion)) {
    return null;
  }
  return { chosenAction, topSuggestion };
}

function updateFeedbackPanel() {
  const summary = document.getElementById("ai-summary");
  const panel = document.getElementById("feedback-panel");
  const text = document.getElementById("feedback-text");
  const status = document.getElementById("feedback-status");
  status.textContent = state.feedbackStatus;

  if (!state.aiContext || !state.aiContext.suggestions.length) {
    summary.textContent = "Ladda AI-förslag för att se hur agenten resonerar.";
    panel.classList.add("hidden");
    text.textContent = "";
    return;
  }

  const source = rankingSource() === "model" ? "modellen" : "search-läraren";
  const top = state.aiContext.suggestions[0];
  const searchPart = top.search_score === null || top.search_score === undefined ? "" : ` | search ${top.search_score.toFixed(1)}`;
  const modelPart = top.model_score === null || top.model_score === undefined ? "" : ` | modell ${top.model_score.toFixed(2)}`;
  summary.textContent = `Aktiv AI-källa: ${source}. Toppförslag: ${top.description}${searchPart}${modelPart}.`;

  const candidate = currentCorrectionCandidate();
  if (!candidate) {
    panel.classList.add("hidden");
    text.textContent = "";
    return;
  }

  panel.classList.remove("hidden");
  text.textContent = `Du håller på att spela ${candidate.chosenAction.description} i stället för AI:ns toppförslag ${candidate.topSuggestion.description}.`;
}

function updateStatus() {
  const snapshot = state.snapshot;
  document.getElementById("status-text").textContent = snapshot.status;
  document.getElementById("stock-count").textContent = snapshot.stock_count;
  document.getElementById("sequence-count").textContent = snapshot.completed_sequences;
  document.getElementById("move-count").textContent = snapshot.moves_played;
  const selection = state.selection;
  document.getElementById("selection-text").textContent = selection
    ? `Vald sekvens: kolumn ${selection.fromColumn + 1}, längd ${selection.runLength}. Klicka eller släpp på målkolumn.`
    : "Ingen flytt vald.";
  updateFeedbackPanel();
}

function renderSuggestions() {
  const container = document.getElementById("suggestions");
  if (!state.suggestions.length) {
    container.className = "suggestions empty";
    container.textContent = "Inga AI-förslag laddade.";
    return;
  }
  container.className = "suggestions";
  container.innerHTML = state.suggestions
    .map((suggestion) => {
      const sourceBadge = state.aiContext && state.aiContext.modelEligible ? "modell" : "search";
      const searchPart =
        suggestion.search_score === null || suggestion.search_score === undefined
          ? ""
          : ` | search ${suggestion.search_score.toFixed(1)}`;
      const modelPart =
        suggestion.model_score === null || suggestion.model_score === undefined
          ? ""
          : ` | modell ${suggestion.model_score.toFixed(3)}`;
      return `<button class="suggestion-item" data-type="${suggestion.type}" data-from="${suggestion.from_column ?? ""}" data-to="${suggestion.to_column ?? ""}" data-run="${suggestion.run_length ?? ""}">
        <span class="suggestion-main">${suggestion.description}</span>
        <span class="suggestion-meta">${sourceBadge} | heuristik ${suggestion.heuristic_score.toFixed(1)}${searchPart}${modelPart}</span>
      </button>`;
    })
    .join("");

  container.querySelectorAll(".suggestion-item").forEach((button) => {
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

function renderBoard() {
  const board = document.getElementById("board");
  board.innerHTML = "";

  state.snapshot.columns.forEach((column, columnIndex) => {
    const columnElement = document.createElement("section");
    columnElement.className = "column";
    columnElement.dataset.columnIndex = String(columnIndex);
    columnElement.innerHTML = `<div class="column-title">Kolumn ${columnIndex + 1}</div>`;

    const stack = document.createElement("div");
    stack.className = `stack ${column.cards.length === 0 ? "empty" : ""}`;
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

        cardElement.className = `card ${card.color} ${movable ? "movable" : ""} ${selected ? "selected" : ""}`;
        cardElement.innerHTML = cardMarkup(card);
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
}

function render() {
  updateStatus();
  renderSuggestions();
  renderBoard();
}

function applySnapshot(snapshot) {
  state.snapshot = snapshot;
  state.gameId = snapshot.game_id;
  state.selection = null;
  state.selectionTarget = null;
  render();
}

async function startNewGame() {
  const snapshot = await api("/game/new", {
    method: "POST",
    body: JSON.stringify({}),
  });
  state.suggestions = [];
  state.aiContext = null;
  state.feedbackStatus = "";
  state.savedFeedbackKey = null;
  applySnapshot(snapshot);
}

async function saveCorrection(chosenAction, note = "human correction") {
  if (!state.aiContext || !state.aiContext.suggestions.length) {
    return;
  }
  const topSuggestion = state.aiContext.suggestions[0];
  if (actionKey(chosenAction) === actionKey(topSuggestion)) {
    return;
  }
  const feedbackKey = `${state.snapshot.state_hash}:${actionKey(chosenAction)}`;
  if (state.savedFeedbackKey === feedbackKey) {
    return;
  }
  const payload = await api("/ai/feedback", {
    method: "POST",
    body: JSON.stringify({
      game_id: state.gameId,
      state_hash: state.snapshot.state_hash,
      ai_source: rankingSource(),
      model_loaded: state.aiContext.modelLoaded,
      model_eligible: state.aiContext.modelEligible,
      recommended_action: topSuggestion,
      chosen_action: chosenAction,
      candidate_actions: state.aiContext.suggestions,
      state_snapshot: state.snapshot,
      note,
    }),
  });
  state.feedbackStatus = `Korrigering sparad i ${payload.path}`;
  state.savedFeedbackKey = feedbackKey;
}

async function performMove(fromColumn, toColumn, runLength) {
  if (state.snapshot.status !== "in_progress") {
    return;
  }
  if (state.aiContext && state.aiContext.stateHash === state.snapshot.state_hash) {
    await saveCorrection(chosenMoveAction(fromColumn, toColumn, runLength));
  }
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
  state.savedFeedbackKey = null;
  applySnapshot(snapshot);
}

async function performDeal() {
  if (state.aiContext && state.aiContext.stateHash === state.snapshot.state_hash) {
    await saveCorrection({ type: "deal", description: `Deal from stock (${state.snapshot.stock_count} left)` }, "human chose deal");
  }
  const snapshot = await api("/game/deal", {
    method: "POST",
    body: JSON.stringify({ game_id: state.gameId }),
  });
  state.suggestions = [];
  state.aiContext = null;
  state.savedFeedbackKey = null;
  applySnapshot(snapshot);
}

async function performUndo() {
  const snapshot = await api("/game/undo", {
    method: "POST",
    body: JSON.stringify({ game_id: state.gameId }),
  });
  state.suggestions = [];
  state.aiContext = null;
  state.savedFeedbackKey = null;
  applySnapshot(snapshot);
}

async function performConcede() {
  const snapshot = await api("/game/concede", {
    method: "POST",
    body: JSON.stringify({ game_id: state.gameId }),
  });
  state.suggestions = [];
  state.aiContext = null;
  state.savedFeedbackKey = null;
  applySnapshot(snapshot);
}

async function loadSuggestions() {
  const payload = await api("/ai/evaluate-move", {
    method: "POST",
    body: JSON.stringify({ game_id: state.gameId }),
  });
  state.suggestions = payload.suggestions;
  state.aiContext = {
    stateHash: state.snapshot.state_hash,
    suggestions: payload.suggestions,
    modelLoaded: payload.model_loaded,
    modelEligible: payload.model_eligible,
  };
  state.feedbackStatus = "";
  state.savedFeedbackKey = null;
  render();
}

async function saveSelectedCorrection() {
  const candidate = currentCorrectionCandidate();
  if (!candidate) {
    return;
  }
  await saveCorrection(candidate.chosenAction, "manual human correction");
  updateFeedbackPanel();
}

document.getElementById("new-game").addEventListener("click", startNewGame);
document.getElementById("deal").addEventListener("click", performDeal);
document.getElementById("undo").addEventListener("click", performUndo);
document.getElementById("suggest").addEventListener("click", loadSuggestions);
document.getElementById("concede").addEventListener("click", performConcede);
document.getElementById("save-feedback").addEventListener("click", saveSelectedCorrection);

startNewGame().catch((error) => {
  document.getElementById("status-text").textContent = error.message;
});
