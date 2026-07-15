/**
 * script.js
 * ----------
 * Small, dependency-free client-side behaviours:
 *  - dark/light theme toggle persisted in localStorage
 *  - animated spectrum meter fill (prediction result page)
 *  - animated bar-chart fills (metrics page)
 *  - basic client-side required-field highlighting (server remains source of truth)
 */

(function () {
  const root = document.documentElement;
  const THEME_KEY = "spotify-popularity-theme";

  function applyTheme(theme) {
    root.setAttribute("data-theme", theme);
    localStorage.setItem(THEME_KEY, theme);
  }

  function initTheme() {
    const stored = localStorage.getItem(THEME_KEY);
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    applyTheme(stored || (prefersDark ? "dark" : "light"));
  }

  function bindThemeToggle() {
    const toggle = document.getElementById("theme-switch");
    if (!toggle) return;
    toggle.addEventListener("click", () => {
      const current = root.getAttribute("data-theme");
      applyTheme(current === "dark" ? "light" : "dark");
    });
  }

  function animateMeter() {
    const fill = document.querySelector(".meter-fill");
    if (!fill) return;
    const target = fill.dataset.value || 0;
    requestAnimationFrame(() => {
      setTimeout(() => {
        fill.style.width = target + "%";
      }, 120);
    });
  }

  function animateBars() {
    const bars = document.querySelectorAll(".bar-fill");
    bars.forEach((bar, idx) => {
      const target = bar.dataset.value || 0;
      setTimeout(() => {
        bar.style.width = target + "%";
      }, 100 + idx * 60);
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    initTheme();
    bindThemeToggle();
    animateMeter();
    animateBars();
  });
})();
