document.addEventListener("DOMContentLoaded", () => {
  // Mark page loaded for hero animations
  document.body.classList.add("page-loaded");

  // --- 1. WebSocket demo logic ---
  const WS_URL = "ws://localhost:8000/ws/query";

  const btnAsk = document.getElementById("btnAsk");
  const inputQuestion = document.getElementById("userQuestion");
  const inputTenant = document.getElementById("tenantId");
  const answerContainer = document.getElementById("answerContainer");
  const sourcesContainer = document.getElementById("sourcesContainer");
  const statusLabel = document.getElementById("statusLabel");

  let socket = null;
  let isStreaming = false;

  function setUIState(state) {
    if (!statusLabel || !btnAsk) return;
    statusLabel.className = "demo-panel__status-label";

    switch (state) {
      case "idle":
        btnAsk.disabled = false;
        btnAsk.textContent = "Hỏi trợ lý";
        statusLabel.textContent = "Sẵn sàng";
        statusLabel.classList.add("status-idle");
        isStreaming = false;
        break;
      case "streaming":
        btnAsk.disabled = true;
        btnAsk.textContent = "Đang xử lý…";
        statusLabel.textContent = "Đang trả lời…";
        statusLabel.classList.add("status-streaming");
        isStreaming = true;
        break;
      case "complete":
        btnAsk.disabled = false;
        btnAsk.textContent = "Hỏi câu khác";
        statusLabel.textContent = "Hoàn tất";
        statusLabel.classList.add("status-complete");
        isStreaming = false;
        break;
      case "error":
        btnAsk.disabled = false;
        btnAsk.textContent = "Thử lại";
        statusLabel.textContent = "Lỗi";
        statusLabel.classList.add("status-error");
        isStreaming = false;
        break;
      default:
        break;
    }
  }

  function handleError(msg) {
    if (answerContainer) {
      answerContainer.innerHTML = `<span style="color: #fed7aa;">⚠️ ${msg}</span>`;
    }
    setUIState("error");
    if (socket) socket.close();
  }

  function handleServerMessage(message) {
    if (!answerContainer || !sourcesContainer) return;

    if (message.type === "meta") {
      sourcesContainer.innerHTML = "";
      if (Array.isArray(message.sources)) {
        message.sources.forEach((s) => {
          const badge = document.createElement("span");
          badge.className = "demo-source-badge";
          badge.textContent = String(s);
          sourcesContainer.appendChild(badge);
        });
      }
      return;
    }

    if (message.type === "chunk") {
      // Remove placeholder on first chunk
      const placeholder = answerContainer.querySelector(".demo-output__placeholder");
      if (placeholder) {
        placeholder.remove();
      }
      answerContainer.textContent += message.text || "";
      answerContainer.scrollTop = answerContainer.scrollHeight;
      return;
    }

    if (message.type === "end") {
      setUIState("complete");
      if (socket) socket.close();
      return;
    }

    if (message.type === "error") {
      handleError(message.text || "Đã xảy ra lỗi không xác định.");
    }
  }

  if (btnAsk) {
    btnAsk.addEventListener("click", (e) => {
      e.preventDefault();
      if (!inputQuestion || !inputTenant) return;

      const question = inputQuestion.value.trim();
      const tenantId = inputTenant.value.trim() || null;

      if (!question) {
        alert("Vui lòng nhập câu hỏi để trải nghiệm demo.");
        return;
      }

      if (answerContainer) {
        answerContainer.textContent = "";
      }
      if (sourcesContainer) {
        sourcesContainer.innerHTML = "";
      }
      setUIState("streaming");

      try {
        socket = new WebSocket(WS_URL);

        socket.onopen = () => {
          const payload = {
            question: question,
            tenant_id: tenantId,
            history: [],
          };
          socket.send(JSON.stringify(payload));
        };

        socket.onmessage = (event) => {
          try {
            const msg = JSON.parse(event.data);
            handleServerMessage(msg);
          } catch (err) {
            console.error("JSON parse error:", err);
          }
        };

        socket.onerror = (err) => {
          console.error("WebSocket error:", err);
          handleError("Lỗi kết nối tới server demo.");
        };

        socket.onclose = () => {
          if (isStreaming) {
            setUIState("idle");
          }
        };
      } catch (err) {
        handleError("Không thể khởi tạo kết nối WebSocket.");
      }
    });
  }

  // --- 2. Smooth scroll for nav links ---
  document.querySelectorAll('a[href^="#"]').forEach((link) => {
    link.addEventListener("click", (e) => {
      const targetId = link.getAttribute("href");
      if (!targetId || targetId === "#") return;
      const target = document.querySelector(targetId);
      if (!target) return;
      e.preventDefault();
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });

  // --- 3. Scroll-based reveal animations ---
  const revealEls = document.querySelectorAll(".js-reveal");
  if ("IntersectionObserver" in window && revealEls.length > 0) {
    const revealObserver = new IntersectionObserver(
      (entries, obs) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            obs.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.18 }
    );
    revealEls.forEach((el) => revealObserver.observe(el));
  } else {
    revealEls.forEach((el) => el.classList.add("is-visible"));
  }
});

