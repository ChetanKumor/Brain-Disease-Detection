/* Progressive enhancement for the Brain Disease Detection UI.
   The form works without JavaScript (server-rendered POST); this script layers
   on a theme toggle, an image preview, and an inline AJAX prediction flow. */
(function () {
  "use strict";

  /* ---- Theme toggle (persisted) ---------------------------------------- */
  var root = document.documentElement;
  var toggle = document.getElementById("theme-toggle");
  var stored = localStorage.getItem("theme");
  if (stored) {
    root.setAttribute("data-theme", stored);
  } else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
    root.setAttribute("data-theme", "dark");
  }
  function syncToggleIcon() {
    var icon = toggle && toggle.querySelector(".theme-toggle-icon");
    if (icon) icon.textContent = root.getAttribute("data-theme") === "dark" ? "☀️" : "🌙";
  }
  syncToggleIcon();
  if (toggle) {
    toggle.addEventListener("click", function () {
      var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      localStorage.setItem("theme", next);
      syncToggleIcon();
    });
  }

  /* ---- Image dropzone + preview ---------------------------------------- */
  var input = document.getElementById("image");
  var dropzone = document.getElementById("dropzone");
  var preview = document.getElementById("preview");
  var dropText = document.getElementById("dropzone-text");

  function showPreview(file) {
    if (!file || !preview) return;
    var reader = new FileReader();
    reader.onload = function (e) {
      preview.src = e.target.result;
      preview.hidden = false;
      if (dropText) dropText.hidden = true;
    };
    reader.readAsDataURL(file);
  }

  if (input) {
    input.addEventListener("change", function () {
      if (input.files && input.files[0]) showPreview(input.files[0]);
    });
  }
  if (dropzone) {
    ["dragenter", "dragover"].forEach(function (ev) {
      dropzone.addEventListener(ev, function (e) {
        e.preventDefault();
        dropzone.classList.add("dragover");
      });
    });
    ["dragleave", "drop"].forEach(function (ev) {
      dropzone.addEventListener(ev, function (e) {
        e.preventDefault();
        dropzone.classList.remove("dragover");
      });
    });
    dropzone.addEventListener("drop", function (e) {
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
        input.files = e.dataTransfer.files;
        showPreview(e.dataTransfer.files[0]);
      }
    });
  }

  /* ---- AJAX prediction -------------------------------------------------- */
  var form = document.getElementById("predict-form");
  var spinner = document.getElementById("spinner");
  var submitBtn = form ? form.querySelector(".btn-primary") : null;
  var errorPanel = document.getElementById("error-panel");
  var errorMessage = document.getElementById("error-message");
  var resultPanel = document.getElementById("result-panel");

  function setLoading(loading) {
    if (spinner) spinner.hidden = !loading;
    if (submitBtn) submitBtn.disabled = loading;
  }
  function showError(message) {
    if (resultPanel) resultPanel.hidden = true;
    if (errorMessage) errorMessage.textContent = message;
    if (errorPanel) errorPanel.hidden = false;
  }
  function pct(x) { return (x * 100).toFixed(1) + "%"; }

  function renderResult(data) {
    if (errorPanel) errorPanel.hidden = true;
    document.getElementById("result-task").textContent = data.display_name;
    document.getElementById("result-label").textContent = data.predicted_label;
    document.getElementById("confidence-value").textContent = pct(data.confidence);
    document.getElementById("confidence-bar").style.width = pct(data.confidence);

    var list = document.getElementById("probabilities");
    list.innerHTML = "";
    data.probabilities
      .slice()
      .sort(function (a, b) { return b.probability - a.probability; })
      .forEach(function (p) {
        var li = document.createElement("li");
        var header = document.createElement("div");
        header.className = "prob-header";
        header.innerHTML = "<span></span><span></span>";
        header.children[0].textContent = p.label;
        header.children[1].textContent = pct(p.probability);
        var bar = document.createElement("div");
        bar.className = "bar";
        var fill = document.createElement("div");
        fill.className = "bar-fill";
        fill.style.width = pct(p.probability);
        bar.appendChild(fill);
        li.appendChild(header);
        li.appendChild(bar);
        list.appendChild(li);
      });
    if (resultPanel) resultPanel.hidden = false;
    resultPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  if (form) {
    form.addEventListener("submit", function (e) {
      e.preventDefault();
      setLoading(true);
      fetch("/api/predict", { method: "POST", body: new FormData(form) })
        .then(function (response) {
          return response.json().then(function (body) {
            return { ok: response.ok, body: body };
          });
        })
        .then(function (res) {
          if (res.ok) renderResult(res.body);
          else showError(res.body.error || "Prediction failed.");
        })
        .catch(function () {
          showError("Could not reach the server. Please try again.");
        })
        .finally(function () { setLoading(false); });
    });
  }
})();
