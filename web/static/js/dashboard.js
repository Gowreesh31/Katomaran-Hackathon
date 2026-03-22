/**
 * dashboard.js
 * Katomaran FaceTrack — Additional interactivity and micro-animations
 */

document.addEventListener("DOMContentLoaded", () => {
  // ── Animate stat cards on load ──────────────────────────────────────────────
  document.querySelectorAll(".stat-card").forEach((card, i) => {
    card.style.opacity = "0";
    card.style.transform = "translateY(20px)";
    card.style.transition = `opacity 0.4s ease ${i * 0.08}s, transform 0.4s ease ${i * 0.08}s`;
    requestAnimationFrame(() => {
      card.style.opacity = "1";
      card.style.transform = "translateY(0)";
    });
  });

  // ── Animate glass cards ─────────────────────────────────────────────────────
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.1 }
  );

  document.querySelectorAll(".glass-card").forEach((card) => {
    card.style.opacity = "0";
    card.style.transform = "translateY(16px)";
    card.style.transition = "opacity 0.45s ease, transform 0.45s ease";
    observer.observe(card);
  });

  document.querySelectorAll(".glass-card.visible").forEach((card) => {
    card.style.opacity = "1";
    card.style.transform = "translateY(0)";
  });

  // Polyfill for browsers without intersection observer
  setTimeout(() => {
    document.querySelectorAll(".glass-card").forEach((card) => {
      card.style.opacity = "1";
      card.style.transform = "translateY(0)";
    });
  }, 600);

  // ── Counter animation for stat values ───────────────────────────────────────
  function animateCounter(el, target, duration = 1200) {
    const startVal = 0;
    const startTime = performance.now();
    const isFloat = target % 1 !== 0;

    function step(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
      const current = startVal + eased * (target - startVal);
      el.textContent = isFloat ? current.toFixed(1) : Math.round(current);
      if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  document.querySelectorAll(".stat-value").forEach((el) => {
    const raw = parseFloat(el.textContent.trim());
    if (!isNaN(raw)) {
      el.textContent = "0";
      setTimeout(() => animateCounter(el, raw), 300);
    }
  });

  // ── Table row hover glow ─────────────────────────────────────────────────────
  document.querySelectorAll(".visitor-table tbody tr").forEach((row) => {
    row.addEventListener("mouseenter", () => {
      row.style.boxShadow = "inset 0 0 0 1px rgba(99,179,237,0.08)";
    });
    row.addEventListener("mouseleave", () => {
      row.style.boxShadow = "";
    });
  });

  // ── Keyboard shortcut: '/' to focus search ───────────────────────────────────
  const searchInput = document.getElementById("searchInput");
  if (searchInput) {
    document.addEventListener("keydown", (e) => {
      if (e.key === "/" && document.activeElement !== searchInput) {
        e.preventDefault();
        searchInput.focus();
      }
      if (e.key === "Escape" && document.activeElement === searchInput) {
        searchInput.blur();
        searchInput.value = "";
        searchInput.dispatchEvent(new Event("input"));
      }
    });
  }

  // ── Live status indicator ────────────────────────────────────────────────────
  async function pingHealth() {
    const dot = document.getElementById("statusDot");
    const txt = document.getElementById("statusText");
    try {
      const res = await fetch("/health", { cache: "no-store" });
      if (res.ok) {
        if (dot) dot.style.background = "#48c78e";
        if (txt) txt.textContent = "LIVE";
      } else {
        if (dot) dot.style.background = "#f6ad55";
        if (txt) txt.textContent = "DEGRADED";
      }
    } catch {
      if (dot) { dot.style.background = "#f56565"; dot.classList.remove("pulse"); }
      if (txt) txt.textContent = "OFFLINE";
    }
  }

  pingHealth();
  setInterval(pingHealth, 15000);
});
