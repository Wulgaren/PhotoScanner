(() => {
  const boards = document.getElementById("boards");
  const thresholdPill = document.getElementById("threshold-pill");
  const progressPill = document.getElementById("progress-pill");
  const deletePill = document.getElementById("delete-pill");
  const btnUndo = document.getElementById("btn-undo");
  const btnFinish = document.getElementById("btn-finish");
  const modal = document.getElementById("modal");
  const modalTitle = document.getElementById("modal-title");
  const modalBody = document.getElementById("modal-body");
  const modalPrimary = document.getElementById("modal-primary");
  const modalSecondary = document.getElementById("modal-secondary");

  /** @type {any} */
  let state = null;
  /** Local toggle overrides: series_id -> Set of delete indices */
  const marks = new Map();
  let focusIdx = 0; // slot index (always 0 when SLOTS=1)
  let photoFocus = 0; // index into series.photos order
  let modalMode = null; // 'tier' | 'finish' | 'done' | null
  let busy = false;
  let builtSignature = "";

  async function api(path, opts = {}) {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...opts,
    });
    return res.json();
  }

  function slotCount() {
    return state?.slots?.length || 1;
  }

  function slotSeries(i) {
    if (!state || !state.slots) return null;
    return state.slots[i] || null;
  }

  function focusedSeries() {
    return slotSeries(focusIdx);
  }

  function focusedPhoto() {
    const series = focusedSeries();
    if (!series || !series.photos.length) return null;
    const i = Math.max(0, Math.min(photoFocus, series.photos.length - 1));
    return series.photos[i] || null;
  }

  function slotsSignature() {
    if (!state) return "";
    return state.slots
      .map((s) => (s ? `${s.series_id}:${s.photos.length}` : "-"))
      .join("|");
  }

  function ensureMarks(series) {
    if (!series) return new Set();
    if (!marks.has(series.series_id)) {
      const initial = new Set(
        series.photos.filter((p) => p.marked_delete).map((p) => p.index)
      );
      marks.set(series.series_id, initial);
    }
    return marks.get(series.series_id);
  }

  function pruneMarks() {
    if (!state) return;
    const live = new Set(
      state.slots.filter(Boolean).map((s) => s.series_id)
    );
    for (const sid of [...marks.keys()]) {
      if (!live.has(sid)) marks.delete(sid);
    }
  }

  function clampPhotoFocus() {
    const series = focusedSeries();
    if (!series || !series.photos.length) {
      photoFocus = 0;
      return;
    }
    if (photoFocus < 0) photoFocus = 0;
    if (photoFocus >= series.photos.length) photoFocus = series.photos.length - 1;
  }

  function applyState(next) {
    const prevSid = focusedSeries()?.series_id;
    state = next;
    pruneMarks();
    for (const s of state.slots) {
      if (s) ensureMarks(s);
    }
    if (!slotSeries(focusIdx)) {
      const first = state.slots.findIndex(Boolean);
      focusIdx = first >= 0 ? first : 0;
    }
    const sid = focusedSeries()?.series_id;
    if (sid !== prevSid) photoFocus = 0;
    clampPhotoFocus();
    render();
    maybeShowTierModal();
  }

  function renderMeta() {
    if (!state) return;
    const p = state.progress || {};
    thresholdPill.textContent = `T ${state.threshold.toFixed(1)}`;
    progressPill.textContent = `${p.done || 0} / ${p.total || 0} · ${p.remaining || 0} left`;
    deletePill.textContent = `${p.confirmed_delete_count || 0} marked`;
    btnUndo.disabled = !state.can_undo || busy;
  }

  function badgeHtml(photo, isDel) {
    const status = isDel
      ? `<span class="status delete">DELETE</span>`
      : `<span class="status keep">KEEP</span>`;
    const best = photo.is_best ? `<span class="lock">BEST</span>` : "";
    return `
      <span class="num">${photo.index + 1}</span>
      <span class="score">${photo.score != null ? photo.score.toFixed(3) : "—"}</span>
      ${status}${best}
    `;
  }

  function setFocus(idx) {
    focusIdx = idx;
    boards.querySelectorAll(".stack").forEach((el) => {
      const slot = Number(el.dataset.slot);
      el.classList.toggle("focused", slot === focusIdx);
    });
    clampPhotoFocus();
    updatePhotoFocusUI(false);
  }

  function setPhotoFocus(photoListIdx, { scroll = true } = {}) {
    photoFocus = photoListIdx;
    clampPhotoFocus();
    updatePhotoFocusUI(scroll);
  }

  function updatePhotoFocusUI(scroll = false) {
    const series = focusedSeries();
    const stack = boards.querySelector(`.stack[data-slot="${focusIdx}"]`);
    if (!stack || !series) return;
    const focused = series.photos[photoFocus];
    const focusedIndex = focused ? focused.index : -1;
    stack.querySelectorAll(".photo").forEach((card) => {
      const on = Number(card.dataset.index) === focusedIndex;
      card.classList.toggle("photo-focused", on);
      if (on && scroll) {
        card.scrollIntoView({ behavior: "smooth", inline: "nearest", block: "nearest" });
      }
    });
  }

  function updateSlotMarks(slotIdx) {
    const series = slotSeries(slotIdx);
    const stack = boards.querySelector(`.stack[data-slot="${slotIdx}"]`);
    if (!series || !stack) return;
    const markSet = ensureMarks(series);
    const headRight = stack.querySelector(".stack-head span:last-child");
    if (headRight) headRight.textContent = `${markSet.size} del`;

    stack.querySelectorAll(".photo").forEach((card) => {
      const index = Number(card.dataset.index);
      const photo = series.photos.find((p) => p.index === index);
      if (!photo) return;
      const isDel = markSet.has(index);
      card.classList.toggle("marked-delete", isDel);
      card.classList.toggle("keep", !isDel);
      card.classList.toggle("is-best", !!photo.is_best);
      const row = card.querySelector(".badge-row");
      if (row) row.innerHTML = badgeHtml(photo, isDel);
    });
    if (slotIdx === focusIdx) updatePhotoFocusUI(false);
  }

  function togglePhotoMark(photo) {
    const series = focusedSeries();
    if (!series || !photo) return;
    const set = ensureMarks(series);
    if (set.has(photo.index)) set.delete(photo.index);
    else set.add(photo.index);
    updateSlotMarks(focusIdx);
  }

  function movePhotoFocus(delta) {
    const series = focusedSeries();
    if (!series || !series.photos.length) return;
    setPhotoFocus(photoFocus + delta);
  }

  function buildBoards() {
    boards.innerHTML = "";
    builtSignature = slotsSignature();

    const n = slotCount();
    for (let i = 0; i < n; i++) {
      const series = state.slots[i];
      const row = document.createElement("section");
      row.className = "stack" + (i === focusIdx ? " focused" : "");
      row.dataset.slot = String(i);

      if (!series) {
        row.classList.add("empty");
        row.textContent = "—";
        row.addEventListener("click", () => setFocus(i));
        boards.appendChild(row);
        continue;
      }

      const markSet = ensureMarks(series);
      const head = document.createElement("div");
      head.className = "stack-head";
      head.innerHTML = `<span>Series ${series.series_id}</span><span>${markSet.size} del</span>`;
      row.appendChild(head);

      const body = document.createElement("div");
      body.className = "stack-body";

      for (const photo of series.photos) {
        const card = document.createElement("article");
        const isDel = markSet.has(photo.index);
        card.className =
          "photo" +
          (isDel ? " marked-delete" : " keep") +
          (photo.is_best ? " is-best" : "");
        card.dataset.index = String(photo.index);

        const img = document.createElement("img");
        img.loading = "lazy";
        img.alt = photo.filename || `photo ${photo.index + 1}`;
        const thumbQs = new URLSearchParams({ path: photo.path });
        if (photo.uuid) thumbQs.set("uuid", photo.uuid);
        img.src = `/thumb?${thumbQs.toString()}`;
        card.appendChild(img);

        const badges = document.createElement("div");
        badges.className = "badge-row";
        badges.innerHTML = badgeHtml(photo, isDel);
        card.appendChild(badges);

        card.addEventListener("click", (ev) => {
          setFocus(i);
          const listIdx = series.photos.findIndex((p) => p.index === photo.index);
          if (listIdx >= 0) setPhotoFocus(listIdx, { scroll: false });
          if (ev.altKey || ev.metaKey) {
            api("/api/preview", {
              method: "POST",
              body: JSON.stringify({ path: photo.path, uuid: photo.uuid }),
            });
            return;
          }
          togglePhotoMark(photo);
        });

        card.addEventListener("contextmenu", (ev) => {
          ev.preventDefault();
          setFocus(i);
          const listIdx = series.photos.findIndex((p) => p.index === photo.index);
          if (listIdx >= 0) setPhotoFocus(listIdx, { scroll: false });
          api("/api/preview", {
            method: "POST",
            body: JSON.stringify({
              path: photo.path,
              uuid: photo.uuid,
              prefer_photos: true,
            }),
          });
        });

        card.addEventListener("dblclick", (ev) => {
          ev.preventDefault();
          api("/api/preview", {
            method: "POST",
            body: JSON.stringify({ path: photo.path, uuid: photo.uuid }),
          });
        });

        body.appendChild(card);
      }

      row.appendChild(body);
      row.addEventListener("click", (ev) => {
        if (ev.target.closest(".photo")) return;
        setFocus(i);
      });
      boards.appendChild(row);
    }
    updatePhotoFocusUI(false);
  }

  function render() {
    if (!state) return;
    renderMeta();
    const sig = slotsSignature();
    if (sig !== builtSignature || !boards.children.length) {
      buildBoards();
    } else {
      setFocus(focusIdx);
      for (let i = 0; i < slotCount(); i++) updateSlotMarks(i);
      updatePhotoFocusUI(false);
    }
  }

  function hideModal() {
    modal.classList.add("hidden");
    modalMode = null;
  }

  function showModal(mode, title, body, primaryLabel, secondaryLabel) {
    modalMode = mode;
    modalTitle.textContent = title;
    modalBody.textContent = body;
    modalPrimary.textContent = primaryLabel;
    modalSecondary.textContent = secondaryLabel;
    modal.classList.remove("hidden");
  }

  function maybeShowTierModal() {
    if (!state || state.finished || modalMode) return;
    if (!state.tier_empty) return;

    if (state.can_raise && state.next_threshold != null) {
      showModal(
        "tier",
        "Tier complete",
        `No series left at threshold ${state.threshold.toFixed(1)}. Raise to ${state.next_threshold.toFixed(1)}? ${state.next_tier_count} new series.`,
        `Raise to ${state.next_threshold.toFixed(1)}`,
        "Finish"
      );
      return;
    }

    showModal(
      "finish",
      "Review complete",
      `No more series up to the threshold cap. ${state.progress.confirmed_delete_count} photos marked for deletion.`,
      "Save & add to album",
      "Save only"
    );
  }

  async function commitFocused() {
    const series = focusedSeries();
    if (!series || busy || state.finished) return;
    busy = true;
    const set = ensureMarks(series);
    const delete_indices = [...set].sort((a, b) => a - b);
    const res = await api("/api/commit", {
      method: "POST",
      body: JSON.stringify({
        series_id: series.series_id,
        delete_indices,
      }),
    });
    busy = false;
    if (!res.ok) {
      alert(res.error || "Commit failed");
      return;
    }
    marks.delete(series.series_id);
    applyState(res.state);
  }

  async function keepAllFocused() {
    const series = focusedSeries();
    if (!series || busy || state.finished) return;
    ensureMarks(series).clear();
    updateSlotMarks(focusIdx);
  }

  async function deleteAllFocused() {
    const series = focusedSeries();
    if (!series || busy || state.finished) return;
    const set = ensureMarks(series);
    set.clear();
    for (const photo of series.photos) set.add(photo.index);
    updateSlotMarks(focusIdx);
  }

  async function undo() {
    if (busy || !state?.can_undo) return;
    busy = true;
    const res = await api("/api/undo", { method: "POST", body: "{}" });
    busy = false;
    if (!res.ok) {
      alert(res.error || "Undo failed");
      return;
    }
    applyState(res.state);
  }

  async function raiseThreshold() {
    busy = true;
    hideModal();
    const res = await api("/api/raise-threshold", {
      method: "POST",
      body: "{}",
    });
    busy = false;
    if (!res.ok) {
      alert(res.error || "Raise failed");
      return;
    }
    applyState(res.state);
  }

  async function finish(addToAlbum) {
    busy = true;
    hideModal();
    const res = await api("/api/finish", {
      method: "POST",
      body: JSON.stringify({ add_to_album: addToAlbum }),
    });
    busy = false;
    if (!res.ok) {
      alert(res.error || "Finish failed");
      return;
    }
    const r = res.result || {};
    const lines = [
      `Marked for deletion: ${r.confirmed_delete_count || 0}`,
      r.output_file ? `Saved: ${r.output_file}` : "No deletions to save.",
    ];
    if (addToAlbum) {
      lines.push(
        r.album_added
          ? `Added ${r.album_count} to “To Delete” album.`
          : "Album update failed or no UUIDs."
      );
    }
    showModal("done", "Session finished", lines.join(" "), "Close", "Add to album");
    if (state) {
      state.finished = true;
      state.finish_result = r;
      renderMeta();
    }
  }

  modalPrimary.addEventListener("click", () => {
    if (modalMode === "tier") raiseThreshold();
    else if (modalMode === "finish") finish(true);
    else if (modalMode === "done") hideModal();
  });

  modalSecondary.addEventListener("click", () => {
    if (modalMode === "tier") finish(false);
    else if (modalMode === "finish") finish(false);
    else if (modalMode === "done") finish(true);
  });

  btnUndo.addEventListener("click", undo);
  btnFinish.addEventListener("click", () => {
    if (state?.finished) {
      const r = state.finish_result || {};
      showModal(
        "done",
        "Session finished",
        `Marked: ${r.confirmed_delete_count || 0}` +
          (r.output_file ? `\n${r.output_file}` : ""),
        "Close",
        "Add to album"
      );
      return;
    }
    showModal(
      "finish",
      "Finish review?",
      `${state?.progress?.confirmed_delete_count || 0} photos currently marked. Save confirmed list now?`,
      "Save & add to album",
      "Save only"
    );
  });

  function toggleDigit(n) {
    const series = focusedSeries();
    if (!series) return;
    const idx = n - 1;
    const listIdx = series.photos.findIndex((p) => p.index === idx);
    if (listIdx < 0) return;
    setPhotoFocus(listIdx);
    togglePhotoMark(series.photos[listIdx]);
  }

  document.addEventListener("keydown", (ev) => {
    if (modalMode && modalMode !== "done") {
      if (ev.key === "Enter") {
        ev.preventDefault();
        modalPrimary.click();
      } else if (ev.key === "Escape") {
        ev.preventDefault();
        if (modalMode === "tier") finish(false);
        else hideModal();
      }
      return;
    }
    if (modalMode === "done") {
      if (ev.key === "Escape" || ev.key === "Enter") {
        ev.preventDefault();
        hideModal();
      }
      return;
    }

    const tag = (ev.target && ev.target.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA") return;

    if (ev.key === "ArrowLeft") {
      ev.preventDefault();
      movePhotoFocus(-1);
      return;
    }
    if (ev.key === "ArrowRight") {
      ev.preventDefault();
      movePhotoFocus(1);
      return;
    }
    if (ev.key === " " || ev.code === "Space") {
      ev.preventDefault();
      togglePhotoMark(focusedPhoto());
      return;
    }
    if (ev.key === "Enter") {
      ev.preventDefault();
      commitFocused();
      return;
    }
    if (ev.key === "s" || ev.key === "S") {
      ev.preventDefault();
      keepAllFocused();
      return;
    }
    if (ev.key === "d" || ev.key === "D") {
      ev.preventDefault();
      deleteAllFocused();
      return;
    }
    if (ev.key === "u" || ev.key === "U") {
      ev.preventDefault();
      undo();
      return;
    }
    if (ev.key >= "1" && ev.key <= "9") {
      ev.preventDefault();
      toggleDigit(Number(ev.key));
    }
  });

  async function boot() {
    const s = await api("/api/state");
    applyState(s);
  }

  boot().catch((err) => {
    boards.textContent = "Failed to load review state. Is the server running?";
    console.error(err);
  });
})();
