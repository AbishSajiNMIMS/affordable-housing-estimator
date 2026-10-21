// Small helpers for the estimator page: a PCA scatter plot drawn on a
// <canvas> (no chart library needed) and a "working..." state on submit.
(function () {
  "use strict";

  var COLOURS = ["#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#6d597a", "#264653"];

  function drawPca() {
    var canvas = document.getElementById("pca-canvas");
    var dataEl = document.getElementById("pca-data");
    if (!canvas || !dataEl) return;
    var data = JSON.parse(dataEl.textContent);
    var points = data.points || [];
    if (!points.length) return;

    // Match the canvas resolution to its displayed size (sharp on phones).
    var ratio = window.devicePixelRatio || 1;
    var width = canvas.clientWidth;
    var height = canvas.clientHeight || 320;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    var ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);

    var all = points.slice();
    if (data.user) all.push([data.user[0], data.user[1], 0]);
    var xs = all.map(function (p) { return p[0]; });
    var ys = all.map(function (p) { return p[1]; });
    var minX = Math.min.apply(null, xs), maxX = Math.max.apply(null, xs);
    var minY = Math.min.apply(null, ys), maxY = Math.max.apply(null, ys);
    var pad = 24;
    function sx(x) { return pad + (x - minX) / (maxX - minX || 1) * (width - 2 * pad); }
    function sy(y) { return height - pad - (y - minY) / (maxY - minY || 1) * (height - 2 * pad); }

    // Axes labels
    ctx.fillStyle = "#6c757d";
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillText("PC1 →", width - pad - 36, height - 6);
    ctx.save();
    ctx.translate(12, pad + 30);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("PC2 →", 0, 0);
    ctx.restore();

    ctx.globalAlpha = 0.55;
    points.forEach(function (p) {
      ctx.fillStyle = COLOURS[(p[2] - 1) % COLOURS.length];
      ctx.beginPath();
      ctx.arc(sx(p[0]), sy(p[1]), 2.5, 0, 2 * Math.PI);
      ctx.fill();
    });
    ctx.globalAlpha = 1;

    if (data.user) {
      drawStar(ctx, sx(data.user[0]), sy(data.user[1]), 9);
    }

    var clusters = Array.from(new Set(points.map(function (p) { return p[2]; }))).sort();
    var legend = document.getElementById("pca-legend");
    if (legend) {
      legend.innerHTML = "Group:" + clusters.map(function (c) {
        return '<span class="legend-dot" style="background:' + COLOURS[(c - 1) % COLOURS.length] + '"></span>' + c;
      }).join("") + (data.user ? '<span class="ms-3">★ your property</span>' : "");
    }
  }

  function drawStar(ctx, x, y, r) {
    ctx.fillStyle = "#000";
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (var i = 0; i < 10; i++) {
      var radius = i % 2 === 0 ? r : r / 2.4;
      var angle = Math.PI / 2 + i * Math.PI / 5;
      ctx.lineTo(x + radius * Math.cos(angle), y - radius * Math.sin(angle));
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }

  document.addEventListener("DOMContentLoaded", function () {
    drawPca();
    var resizeTimer;
    window.addEventListener("resize", function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(drawPca, 150);
    });

    var form = document.getElementById("estimate-form");
    var button = document.getElementById("submit-btn");
    if (form && button) {
      form.addEventListener("submit", function () {
        button.disabled = true;
        button.textContent = "Estimating…";
      });
    }

    var results = document.getElementById("results");
    if (results && window.innerWidth < 992) {
      results.scrollIntoView({ behavior: "smooth" });
    }
  });
})();
