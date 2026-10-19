const $ = (sel, root = document) => root.querySelector(sel);

const thread = $("#thread");
const form = $("#composer");
const input = $("#question");
const sendButton = $("#send");
const uploadStatus = $("#upload-status");

// ---- documents ----

async function loadDocuments() {
  const res = await fetch("/api/documents");
  if (!res.ok) return;
  const docs = await res.json();
  const list = $("#doc-list");
  list.replaceChildren(...docs.map(renderDocument));
  $("#doc-count").textContent = docs.length ? `(${docs.length})` : "";
  $("#empty-docs").hidden = docs.length > 0;
}

function renderDocument(doc) {
  const li = document.createElement("li");
  const ext = doc.file_name.split(".").pop().toUpperCase().slice(0, 4);
  const pages = doc.pages ? `${doc.pages} pages, ` : "";

  const type = el("span", "doc-type", ext);
  const info = el("div");
  info.append(el("div", "doc-name", doc.file_name), el("div", "doc-meta", `${pages}${doc.chunks} chunks`));
  info.firstChild.title = doc.file_name;

  const remove = el("button", "doc-remove", "×");
  remove.type = "button";
  remove.title = `Remove ${doc.file_name}`;
  remove.addEventListener("click", async () => {
    if (!confirm(`Remove ${doc.file_name} from the index?`)) return;
    await fetch(`/api/documents/${encodeURIComponent(doc.file_name)}`, { method: "DELETE" });
    loadDocuments();
  });

  li.append(type, info, remove);
  return li;
}

async function upload(files) {
  for (const file of files) {
    setUploadStatus(`Indexing ${file.name}...`);
    const body = new FormData();
    body.append("file", file);
    const res = await fetch("/api/documents", { method: "POST", body });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      setUploadStatus(`${file.name}: ${data.detail || res.statusText}`, true);
      return;
    }
    setUploadStatus(data.skipped ? `${file.name} was already indexed.` : `Added ${file.name} (${data.chunks} chunks).`);
    await loadDocuments();
  }
}

function setUploadStatus(text, isError = false) {
  uploadStatus.textContent = text;
  uploadStatus.classList.toggle("error", isError);
}

const dropzone = $("#dropzone");
$("#file-input").addEventListener("change", (e) => upload([...e.target.files]).finally(() => (e.target.value = "")));
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragging"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragging"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragging");
  upload([...e.dataTransfer.files]);
});

// ---- asking ----

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (question) ask(question);
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${input.scrollHeight}px`;
});

$("#examples").addEventListener("click", (e) => {
  if (e.target.tagName === "BUTTON") ask(e.target.textContent);
});

async function ask(question) {
  $("#intro")?.remove();
  input.value = "";
  input.style.height = "auto";
  sendButton.disabled = true;

  const message = $("#message-template").content.firstElementChild.cloneNode(true);
  $(".question", message).textContent = question;
  thread.append(message);
  message.scrollIntoView({ block: "start", behavior: "smooth" });

  const answerEl = $(".answer", message);
  let answer = "";
  let sources = [];

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);

    for await (const { event, data } of readEvents(res.body)) {
      if (event === "sources") {
        sources = data;
        answerEl.innerHTML = '<span class="typing">Writing the answer</span>';
      } else if (event === "token") {
        answer += data;
        answerEl.innerHTML = renderAnswer(answer);
      } else if (event === "done") {
        renderSources(message, sources, new Set(data.cited));
      } else if (event === "error") {
        throw new Error(data);
      }
    }
  } catch (err) {
    answerEl.classList.add("error");
    answerEl.textContent = typeof err.message === "string" ? err.message : String(err);
  } finally {
    sendButton.disabled = false;
    input.focus();
  }
}

// Minimal SSE reader for a POST response (EventSource only does GET).
async function* readEvents(stream) {
  const reader = stream.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += value.replace(/\r\n/g, "\n");
    let end;
    while ((end = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      let event = "message";
      const data = [];
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
      }
      if (data.length) yield { event, data: JSON.parse(data.join("\n")) };
    }
  }
}

// ---- rendering ----

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function inline(text) {
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\[(\d+)\]/g, '<button type="button" class="cite" data-n="$1">$1</button>');
}

// Just enough markdown for what the model tends to write: paragraphs, bullet and
// numbered lists, bold. Everything is escaped first.
function renderAnswer(text) {
  const html = [];
  let list = null;
  for (const line of text.split("\n")) {
    const bullet = line.match(/^\s*[-*]\s+(.*)/);
    const numbered = line.match(/^\s*\d+\.\s+(.*)/);
    const item = bullet || numbered;
    const tag = bullet ? "ul" : "ol";
    if (item) {
      if (list !== tag) {
        if (list) html.push(`</${list}>`);
        html.push(`<${tag}>`);
        list = tag;
      }
      html.push(`<li>${inline(item[1])}</li>`);
      continue;
    }
    if (list) { html.push(`</${list}>`); list = null; }
    if (line.trim()) html.push(`<p>${inline(line)}</p>`);
  }
  if (list) html.push(`</${list}>`);
  return html.join("");
}

function renderSources(message, sources, cited) {
  if (!sources.length) return;
  const box = $(".sources", message);
  const list = $(".source-list", message);
  box.hidden = false;

  // Cited sources first, the rest tucked behind a toggle.
  const ordered = [...sources].sort((a, b) => cited.has(b.number) - cited.has(a.number));
  const hidden = [];
  for (const source of ordered) {
    const card = renderSource(source, !cited.size || cited.has(source.number));
    if (cited.size && !cited.has(source.number)) {
      card.hidden = true;
      hidden.push(card);
    }
    list.append(card);
  }

  const more = $(".more-sources", message);
  if (hidden.length) {
    more.hidden = false;
    more.textContent = `Show ${hidden.length} more retrieved but not cited`;
    more.addEventListener("click", () => {
      hidden.forEach((card) => (card.hidden = false));
      more.hidden = true;
    });
  }

  $(".answer", message).addEventListener("click", (e) => {
    const n = e.target.dataset?.n;
    if (!n) return;
    const card = $(`.source[data-n="${n}"]`, message);
    if (!card) return;
    card.hidden = false;
    openSource(card, true);
    card.scrollIntoView({ block: "nearest", behavior: "smooth" });
    card.classList.add("flash");
    setTimeout(() => card.classList.remove("flash"), 1200);
  });
}

function renderSource(source, isCited) {
  const li = el("li", `source${isCited ? "" : " uncited"}`);
  li.dataset.n = source.number;

  const head = el("button", "source-head");
  head.type = "button";
  head.setAttribute("aria-expanded", "false");
  const title = el("span", "source-title", source.file_name);
  if (source.location) title.append(el("span", "where", `  ·  ${source.location}`));
  head.append(
    el("span", "source-num", `[${source.number}]`),
    title,
    el("span", "source-score", `${source.score.toFixed(2)} match`),
  );

  const text = el("blockquote", "source-text", source.text);
  text.hidden = true;

  head.addEventListener("click", () => openSource(li, text.hidden));
  li.append(head, text);
  return li;
}

function openSource(card, open) {
  $(".source-text", card).hidden = !open;
  $(".source-head", card).setAttribute("aria-expanded", String(open));
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

loadDocuments();
input.focus();
