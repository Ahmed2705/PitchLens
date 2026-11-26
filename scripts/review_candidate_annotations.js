"use strict";

// Canvas rendering, frame navigation and pointer-coordinate mapping follow the
// Foundation manual annotation toolkit. Drafts are saved by the local server.
const $ = id => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");
const work = $("work");
const colors = ["#8df977", "#ffac61", "#66e7ff", "#fff"];
const zoomLevels = [1, 1.5, 2, 3, 4, 6, 8];
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
let frames, groups, classes, state, size, image;
let index = 0, selected = -1, mode = "select", addRole = 0, zoom = 0, drag = null;
let ready = false, pending = false, saving = null, exporting = false, saveError = "";
const frame = () => frames[index];
const draft = item => state.drafts[item.id] ?? {status: "unreviewed", boxes: item.boxes};

function notice(message) {
  $("notice").textContent = message;
  $("notice").hidden = !message;
}

async function request(path, payload) {
  const response = await fetch(path, payload === undefined ? undefined : {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload),
  });
  if (!response.headers.get("Content-Type")?.includes("application/json")) {
    throw Error(`Local server returned HTTP ${response.status}.`);
  }
  const result = await response.json();
  if (!response.ok) throw Error(result.error);
  return result;
}

function save() {
  if (saving) return saving;
  if (!pending) return Promise.resolve(true);
  saving = (async () => {
    $("saved").textContent = "Saving…";
    $("retry").hidden = true;
    try {
      while (pending) {
        pending = false;
        const result = await request("/save", state);
        state.revision = result.revision;
      }
      $("saved").textContent = "Progress saved";
      if (saveError && $("notice").textContent === saveError) notice("");
      saveError = "";
      return true;
    } catch (error) {
      pending = true;
      $("saved").textContent = "Unsaved changes";
      $("retry").hidden = false;
      saveError = error.message;
      notice(saveError);
      return false;
    } finally {
      saving = null;
    }
  })();
  return saving;
}

function queueSave() {
  pending = true;
  save();
}

function updateFrame(boxes, status) {
  state.drafts[frame().id] = {boxes, status};
  queueSave();
  refresh();
}

function edit(boxes) {
  const status = draft(frame()).status;
  updateFrame(boxes, status === "reviewed" ? "unreviewed" : status);
}

function fitScale() {
  return Math.min(work.clientWidth, work.clientHeight) / size;
}

function centerOn(x, y) {
  const scale = canvas.clientWidth / size;
  work.scrollLeft = x * scale - work.clientWidth / 2;
  work.scrollTop = y * scale - work.clientHeight / 2;
}

function setZoom(level, target) {
  const scale = canvas.getBoundingClientRect().width / size;
  const center = target ?? (scale ? [
    (work.scrollLeft + work.clientWidth / 2) / scale,
    (work.scrollTop + work.clientHeight / 2) / scale,
  ] : [size / 2, size / 2]);
  zoom = clamp(level, 0, zoomLevels.length - 1);
  canvas.style.width = `${Math.max(1, Math.floor(size * fitScale() * zoomLevels[zoom]))}px`;
  $("zoom-level").value = zoom ? `${zoomLevels[zoom]}× fit` : "Fit";
  $("zoom-out").disabled = zoom === 0;
  $("zoom-in").disabled = zoom === zoomLevels.length - 1;
  centerOn(...center);
  draw();
}

function corners(box) {
  const [, l, t, r, b] = box;
  return [[l, t], [r, t], [l, b], [r, b]];
}

function draw() {
  ctx.clearRect(0, 0, size, size);
  if (!ready) return;
  ctx.drawImage(image, 0, 0);
  const unit = size / Math.max(1, canvas.getBoundingClientRect().width);
  if ($("show-boxes").checked) {
    const boxes = draft(frame()).boxes;
    const paint = (box, active) => {
      const [role, l, t, r, b] = box;
      ctx.strokeStyle = "#14242c";
      ctx.lineWidth = Math.max(1.15, (active ? 4 : 3) * unit);
      ctx.strokeRect(l, t, r - l, b - t);
      ctx.strokeStyle = colors[role];
      ctx.lineWidth = Math.max(.65, (active ? 2.5 : 1.4) * unit);
      ctx.strokeRect(l, t, r - l, b - t);
      if (active) {
        ctx.fillStyle = colors[role];
        for (const [x, y] of corners(box)) {
          ctx.fillRect(x - 3 * unit, y - 3 * unit, 6 * unit, 6 * unit);
        }
      }
    };
    boxes.forEach((box, n) => { if (n !== selected) paint(box, false); });
    if (selected >= 0) paint(drag?.preview ?? boxes[selected], true);
    if (drag?.kind === "add") paint(drag.preview, false);
  }
  if ($("show-flags").checked) {
    frame().flags.forEach(([x, y], n) => {
      ctx.strokeStyle = "#ffcb60";
      ctx.lineWidth = 2 * unit;
      ctx.beginPath();
      ctx.arc(x, y, 9 * unit, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.fillStyle = "#382805";
      ctx.fillRect(x + 9 * unit, y - 17 * unit, 16 * unit, 15 * unit);
      ctx.fillStyle = "#ffe090";
      ctx.font = `${12 * unit}px system-ui`;
      ctx.fillText(n + 1, x + 13 * unit, y - 5 * unit);
    });
  }
}

function refresh() {
  const item = frame();
  const {boxes, status} = draft(item);
  const visible = $("show-boxes").checked;
  $("prev").disabled = index === 0;
  $("next").disabled = index === frames.length - 1;
  $("pending").disabled = !frames.some(other => other !== item && draft(other).status === "unreviewed");
  $("counts").textContent = groups.map((group, n) => {
    const subset = frames.filter(other => other.group === n);
    const reviewed = subset.filter(other => draft(other).status === "reviewed").length;
    const uncertain = subset.filter(other => draft(other).status === "uncertain").length;
    return `${group} ${reviewed}/${subset.length} reviewed${uncertain ? ` · ${uncertain} uncertain` : ""}`;
  }).join("  |  ");
  $("frame").options[index].textContent = optionLabel(item);
  document.querySelectorAll("[data-status]").forEach(button => {
    button.disabled = !ready;
    button.setAttribute("aria-pressed", button.dataset.status === status);
  });
  $("role").value = mode === "select" && selected >= 0 ? boxes[selected][0] : addRole;
  $("remove").disabled = !ready || !visible || selected < 0;
  $("add").disabled = !ready || !visible;
  $("add").setAttribute("aria-pressed", mode === "add");
  $("select").setAttribute("aria-pressed", mode === "select");
  const list = $("boxes");
  const scroll = list.scrollTop;
  list.replaceChildren(...boxes.map(([role], n) => listButton(`${n + 1}. ${classes[role]}`, button => {
    button.disabled = !ready || !visible;
    button.setAttribute("aria-pressed", selected === n);
    button.onclick = () => selectBox(n, true);
  })));
  list.scrollTop = scroll;
  draw();
}

function listButton(label, configure) {
  const li = document.createElement("li");
  const button = document.createElement("button");
  button.textContent = label;
  configure(button);
  li.append(button);
  return li;
}

function optionLabel(item) {
  return `${item.number} · ${draft(item).status}${item.flags.length ? " · flagged" : ""}`;
}

function load(position, persist = true) {
  if (position < 0 || position >= frames.length) return;
  index = position;
  selected = -1;
  mode = "select";
  drag = null;
  ready = false;
  $("frame").selectedIndex = index;
  const item = frame();
  const currentImage = new Image();
  image = currentImage;
  $("flags").replaceChildren(...item.flags.map(([x, y, description], n) => listButton(
    `${n + 1}. ${description}`, button => {
      button.onclick = () => {
        $("show-flags").checked = true;
        setZoom(Math.max(zoom, 3), [x, y]);
      };
    },
  )));
  $("no-flags").hidden = item.flags.length > 0;
  refresh();
  setZoom(0);
  currentImage.onload = () => {
    if (image !== currentImage) return;
    if (image.naturalWidth !== size || image.naturalHeight !== size) {
      notice("This frame has unexpected dimensions.");
      return;
    }
    ready = true;
    if (!saveError) notice("");
    refresh();
  };
  currentImage.onerror = () => {
    if (image === currentImage) notice(`Could not load frame ${item.number}.`);
  };
  currentImage.src = `/image/${item.id}`;
  if (persist) {
    state.cursor = item.id;
    queueSave();
  }
}

function selectBox(n, center = false) {
  selected = n;
  mode = "select";
  refresh();
  if (center && zoom && n >= 0) {
    const [, l, t, r, b] = draft(frame()).boxes[n];
    centerOn((l + r) / 2, (t + b) / 2);
  }
}

function pickRole(role) {
  addRole = role;
  const boxes = draft(frame()).boxes;
  if (ready && mode === "select" && selected >= 0 && boxes[selected][0] !== role) {
    edit(boxes.map((box, n) => n === selected ? [role, ...box.slice(1)] : box));
  } else {
    refresh();
  }
}

function removeBox() {
  if ($("remove").disabled) return;
  const boxes = draft(frame()).boxes.filter((_, n) => n !== selected);
  selected = -1;
  edit(boxes);
}

function pointer(event) {
  const rect = canvas.getBoundingClientRect();
  return [clamp(Math.round((event.clientX - rect.left) * size / rect.width), 0, size),
          clamp(Math.round((event.clientY - rect.top) * size / rect.height), 0, size)];
}

canvas.onpointerdown = event => {
  if (!ready || drag || event.button !== 0 || !$("show-boxes").checked) return;
  const [x, y] = pointer(event);
  const boxes = draft(frame()).boxes;
  if (mode === "add") {
    drag = {kind: "add", x, y, box: [addRole, x, y, x, y]};
  } else {
    const tolerance = 8 * size / canvas.getBoundingClientRect().width;
    if (selected >= 0) {
      const box = boxes[selected];
      const distances = corners(box).map(([cx, cy]) => Math.hypot(cx - x, cy - y));
      const closest = Math.min(...distances);
      if (closest <= tolerance) drag = {kind: "resize", corner: distances.indexOf(closest), x, y, box};
    }
    if (!drag) {
      const hits = boxes.map((box, n) => {
        const [, l, t, r, b] = box;
        return {n, inside: x >= l && x <= r && y >= t && y <= b, area: (r - l) * (b - t)};
      }).filter(hit => hit.inside).sort((a, b) => a.area - b.area);
      const hit = hits.find(candidate => candidate.n === selected) ?? hits[0];
      selectBox(hit?.n ?? -1);
      if (hit) drag = {kind: "move", x, y, box: boxes[selected]};
    }
  }
  if (drag) {
    drag.preview = drag.box;
    drag.pointer = event.pointerId;
    canvas.setPointerCapture(event.pointerId);
  }
};

function preview(event) {
  const [x, y] = pointer(event);
  const [role, l, t, r, b] = drag.box;
  const dx = x - drag.x, dy = y - drag.y;
  if (drag.kind === "add") {
    return [role, Math.min(l, x), Math.min(t, y), Math.max(r, x), Math.max(b, y)];
  }
  if (drag.kind === "move") {
    const mx = clamp(dx, -l, size - r), my = clamp(dy, -t, size - b);
    return [role, l + mx, t + my, r + mx, b + my];
  }
  const right = drag.corner % 2, bottom = drag.corner >= 2;
  return [role, right ? l : clamp(l + dx, 0, Math.max(0, r - 1)),
                bottom ? t : clamp(t + dy, 0, Math.max(0, b - 1)),
                right ? clamp(r + dx, Math.min(size, l + 1), size) : r,
                bottom ? clamp(b + dy, Math.min(size, t + 1), size) : b];
}

canvas.onpointermove = event => {
  if (drag?.pointer !== event.pointerId) return;
  drag.preview = preview(event);
  draw();
};
canvas.onpointerup = event => {
  if (drag?.pointer !== event.pointerId) return;
  const box = preview(event);
  const {kind, x, y} = drag;
  const [endX, endY] = pointer(event);
  drag = null;
  if ((x === endX && y === endY) || box[1] >= box[3] || box[2] >= box[4]) {
    draw();
    return;
  }
  const boxes = draft(frame()).boxes.slice();
  if (kind === "add") {
    selected = boxes.length;
    mode = "select";
    boxes.push(box);
  } else {
    if (box.every((value, n) => value === boxes[selected][n])) {
      draw();
      return;
    }
    boxes[selected] = box;
  }
  edit(boxes);
};
canvas.onpointercancel = event => {
  if (drag?.pointer === event.pointerId) {
    drag = null;
    draw();
  }
};

$("frame").onchange = () => load($("frame").selectedIndex);
$("prev").onclick = () => load(index - 1);
$("next").onclick = () => load(index + 1);
$("pending").onclick = () => {
  const next = frames.findIndex((item, n) => n !== index && draft(item).status === "unreviewed");
  if (next >= 0) load(next);
};
$("fit").onclick = () => setZoom(0);
$("zoom-in").onclick = () => setZoom(zoom + 1);
$("zoom-out").onclick = () => setZoom(zoom - 1);
$("show-boxes").onchange = () => selectBox(-1);
$("show-flags").onchange = draw;
$("add").onclick = () => { selected = -1; mode = "add"; refresh(); };
$("select").onclick = () => selectBox(-1);
$("role").onchange = () => pickRole(Number($("role").value));
$("remove").onclick = removeBox;
$("retry").onclick = save;
document.querySelectorAll("[data-status]").forEach(button => {
  button.onclick = () => {
    if (ready && draft(frame()).status !== button.dataset.status) {
      updateFrame(draft(frame()).boxes, button.dataset.status);
    }
  };
});
$("export").onclick = async () => {
  exporting = true;
  $("export").disabled = true;
  try {
    if ((pending || saving) && !await save()) return;
    const result = await request("/export", {revision: state.revision});
    notice(`Exported reviewed labels to ${result.directory}`);
  } catch (error) {
    notice(error.message);
  } finally {
    exporting = false;
    $("export").disabled = false;
  }
};

document.addEventListener("keydown", event => {
  if (!state || drag || event.altKey || event.ctrlKey || event.metaKey
      || event.target.closest("input, select, textarea")) return;
  const actions = {
    arrowleft: "prev", arrowright: "next", a: "add", escape: "select",
    delete: "remove", "+": "zoom-in", "=": "zoom-in", "-": "zoom-out", "0": "fit",
  };
  const key = event.key.toLowerCase();
  if (actions[key]) {
    event.preventDefault();
    $(actions[key]).click();
  } else if (/^[1-4]$/.test(key)) {
    pickRole(Number(key) - 1);
  } else if (key === "r" || key === "u") {
    document.querySelector(`[data-status=${key === "r" ? "reviewed" : "uncertain"}]`).click();
  }
});
window.addEventListener("beforeunload", event => {
  if (pending || saving || exporting) event.preventDefault();
});
window.addEventListener("resize", () => { if (state) setZoom(zoom); });

async function start() {
  ({frames, groups, classes, state, size} = await request("/state"));
  canvas.width = canvas.height = size;
  groups.forEach((label, n) => {
    const group = document.createElement("optgroup");
    group.label = label;
    frames.filter(item => item.group === n).forEach(item => group.append(new Option(optionLabel(item))));
    $("frame").append(group);
  });
  classes.forEach((label, n) => $("role").add(new Option(label, n)));
  $("editor").hidden = false;
  $("export").disabled = false;
  $("saved").textContent = "Automatic saving enabled";
  load(frames.findIndex(item => item.id === state.cursor), false);
}
start().catch(error => { notice(error.message); $("saved").textContent = "Unable to start"; });
