(() => {
  "use strict";

  /**
   * Second-Look Review Field - AANCA hero canvas.
   * Metaphor: source annotations stay put; review copies join a short queue.
   */

  const CONFIG = {
    seed: 0x52c00d1e,
    // Global pace multiplier (1 = designed timings). Change this to retune the whole story.
    pace: 1,

    desktop: {
      nucleusCount: 56,
      selectCount: 4,
      auditCount: 10,
      queueSlots: 4,
      fieldCx: 0.7,
      fieldCy: 0.52,
      fieldRadiusX: 0.34,
      fieldRadiusY: 0.38,
      nucleusMin: 13,
      nucleusMax: 21,
      queueRight: 0.93,
      queueTop: 0.28,
      queueGap: 52,
    },
    mobile: {
      nucleusCount: 28,
      selectCount: 3,
      auditCount: 7,
      queueSlots: 3,
      fieldCx: 0.5,
      fieldCy: 0.72,
      fieldRadiusX: 0.38,
      fieldRadiusY: 0.22,
      nucleusMin: 9,
      nucleusMax: 14,
      queueRight: 0.5,
      queueTop: 0.9,
      queueGap: 56,
      horizontalQueue: true,
    },
    mobileBreakpoint: 720,

    timing: {
      sceneEnterMs: 750,
      frameTravelMs: [950, 1150],
      frameSettleMs: [200, 240],
      inspectMs: [520, 700],
      markDrawMs: 420,
      copyFlightMs: [1000, 1250],
      queueHoldMs: 800,
      zoomToPatchMs: 2600,
      zoomToStudyMs: 2600,
      finalSettleMs: 1900,
      appearMs: 600,
      fixedStepMs: 1000 / 60,
      maxFrameDeltaMs: 50,
      springSubstepMs: 12,
    },

    camera: {
      // Tuned for dt in seconds with soft ease-in / ease-out, no visible bounce.
      stiffness: 18,
      damping: 9.5,
      settleEpsilon: 0.0012,
    },

    colors: {
      fill: [15, 16, 17],
      nucleusBody: [24, 26, 30],
      annotation: [52, 52, 58],
      annotationSoft: [35, 37, 42],
      selected: [94, 106, 210],
      active: [130, 143, 255],
      queueSlot: [20, 21, 22],
      patchDim: [18, 19, 21],
      patchEdge: [52, 52, 58],
    },
  };

  function mulberry32(seed) {
    let t = seed >>> 0;
    return () => {
      t += 0x6d2b79f5;
      let r = Math.imul(t ^ (t >>> 15), 1 | t);
      r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
      return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
  }

  function clamp(v, a, b) {
    return v < a ? a : v > b ? b : v;
  }

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function easeInOutCubic(t) {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  }

  function easeOutCubic(t) {
    const u = 1 - t;
    return 1 - u * u * u;
  }

  function easeInOutQuint(t) {
    return t < 0.5 ? 16 * t * t * t * t * t : 1 - Math.pow(-2 * t + 2, 5) / 2;
  }

  function cubicBezier(p0, p1, p2, p3, t) {
    const u = 1 - t;
    const tt = t * t;
    const uu = u * u;
    return uu * u * p0 + 3 * uu * t * p1 + 3 * u * tt * p2 + tt * t * p3;
  }

  function rgba(rgb, a) {
    return `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${a})`;
  }

  function pickRange(rng, pair) {
    return lerp(pair[0], pair[1], rng());
  }

  function paced(ms) {
    return ms * CONFIG.pace;
  }

  function buildNucleusPath(rng, rx, ry) {
    const points = 8 + Math.floor(rng() * 5);
    const verts = [];
    for (let i = 0; i < points; i += 1) {
      const a = (i / points) * Math.PI * 2;
      const wobble = 0.82 + rng() * 0.28;
      verts.push({
        x: Math.cos(a) * rx * wobble,
        y: Math.sin(a) * ry * wobble,
      });
    }
    const path = new Path2D();
    const n = verts.length;
    for (let i = 0; i < n; i += 1) {
      const curr = verts[i];
      const next = verts[(i + 1) % n];
      const midX = (curr.x + next.x) * 0.5;
      const midY = (curr.y + next.y) * 0.5;
      if (i === 0) {
        const prev = verts[n - 1];
        path.moveTo((prev.x + curr.x) * 0.5, (prev.y + curr.y) * 0.5);
      }
      path.quadraticCurveTo(curr.x, curr.y, midX, midY);
    }
    path.closePath();
    return path;
  }

  function placeNuclei(layout, rng) {
    const nuclei = [];
    let attempts = 0;
    while (nuclei.length < layout.nucleusCount && attempts < layout.nucleusCount * 80) {
      attempts += 1;
      const ang = rng() * Math.PI * 2;
      const rad = Math.sqrt(rng());
      const nx = Math.cos(ang) * rad;
      const ny = Math.sin(ang) * rad;
      // Soft irregular field (not a perfect ellipse fill).
      const edge = 0.88 + rng() * 0.14;
      if (nx * nx + ny * ny > edge * edge) continue;
      const lx = nx * layout.fieldRadiusX;
      const ly = ny * layout.fieldRadiusY;
      const size = lerp(layout.nucleusMin, layout.nucleusMax, rng());
      const rx = size * (0.78 + rng() * 0.3);
      const ry = size * (0.7 + rng() * 0.32);
      const rot = (rng() - 0.5) * 0.7;
      let ok = true;
      for (let i = 0; i < nuclei.length; i += 1) {
        const o = nuclei[i];
        const d = Math.hypot(lx - o.lx, ly - o.ly);
        const need = (size + o.size) * 0.0011 + 0.012;
        if (d < need) {
          ok = false;
          break;
        }
      }
      if (!ok) continue;
      const path = buildNucleusPath(rng, rx, ry);
      const annPad = 2.2 + rng() * 1.6;
      nuclei.push({
        id: nuclei.length,
        lx,
        ly,
        size,
        rx,
        ry,
        rot,
        path,
        annPad,
        appearAt: 0,
        selected: false,
        marked: false,
        markProgress: 0,
        markStartedAt: -1,
        dim: 1,
      });
    }
    return nuclei;
  }

  function chooseAuditPlan(nuclei, layout, rng) {
    const order = nuclei.map((_, i) => i);
    // Deterministic shuffle.
    for (let i = order.length - 1; i > 0; i -= 1) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = order[i];
      order[i] = order[j];
      order[j] = tmp;
    }
    const audit = order.slice(0, layout.auditCount);
    const selected = [];
    const seen = new Set();
    const step = Math.max(1, Math.floor(audit.length / layout.selectCount));
    for (let i = 0; i < layout.selectCount; i += 1) {
      const idx = Math.min(audit.length - 1, i * step + (i % 2));
      const id = audit[idx];
      if (!seen.has(id)) {
        seen.add(id);
        selected.push(id);
      }
    }
    for (const id of audit) {
      if (selected.length >= layout.selectCount) break;
      if (!seen.has(id)) {
        seen.add(id);
        selected.push(id);
      }
    }
    return { auditOrder: audit, selectIds: new Set(selected) };
  }

  function createController(canvas) {
    const ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) return null;

    const reduced =
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let width = 1;
    let height = 1;
    let dpr = 1;
    let isMobile = false;
    let layout = CONFIG.desktop;
    let fieldScale = 1;

    let nuclei = [];
    let auditOrder = [];
    let selectIds = new Set();
    let queue = [];
    let flights = [];
    let patches = null;

    let phase = "enter";
    let phaseStartedAt = 0;
    let storyTime = 0;
    let auditIndex = -1;
    let finishedOnce = false;

    // Frame motion (continuous)
    let frame = {
      x: 0,
      y: 0,
      size: 20,
      fromX: 0,
      fromY: 0,
      toX: 0,
      toY: 0,
      c1x: 0,
      c1y: 0,
      c2x: 0,
      c2y: 0,
      fromSize: 20,
      toSize: 20,
      startedAt: 0,
      duration: 1,
      visible: false,
      settleUntil: 0,
      inspectUntil: 0,
      activeAccent: 0,
    };

    let camera = { scale: 1, velocity: 0, target: 1, x: 0, y: 0, tx: 0, ty: 0, vx: 0, vy: 0 };
    let nonPriorityDim = 1;
    let studyReveal = 0;

    let raf = 0;
    let running = false;
    let visible = false;
    let pageVisible = !document.hidden;
    let previousFrameTime = 0;
    let accumulator = 0;
    let rng = mulberry32(CONFIG.seed);

    function measure() {
      const rect = canvas.getBoundingClientRect();
      width = Math.max(1, Math.round(rect.width));
      height = Math.max(1, Math.round(rect.height));
      dpr = Math.min(2, window.devicePixelRatio || 1);
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      isMobile = width <= CONFIG.mobileBreakpoint;
      layout = isMobile ? CONFIG.mobile : CONFIG.desktop;
      fieldScale = Math.min(width, height);
    }

    function worldOf(n) {
      return {
        x: width * layout.fieldCx + n.lx * fieldScale,
        y: height * layout.fieldCy + n.ly * fieldScale,
      };
    }

    function queueSlotPos(slotIndex) {
      if (layout.horizontalQueue) {
        const total = layout.queueSlots;
        const span = Math.min(width * 0.72, layout.queueGap * (total - 1) + 40);
        const startX = width * layout.queueRight - span * 0.5;
        return {
          x: startX + (total === 1 ? 0 : (slotIndex / (total - 1)) * span),
          y: height * layout.queueTop,
          scale: slotIndex === 0 ? 0.92 : 0.78,
        };
      }
      return {
        x: width * layout.queueRight,
        y: height * layout.queueTop + slotIndex * layout.queueGap,
        scale: slotIndex === 0 ? 0.95 : 0.8,
      };
    }

    function rebuildScene() {
      rng = mulberry32(CONFIG.seed ^ (isMobile ? 0x1111 : 0));
      nuclei = placeNuclei(layout, rng);
      const plan = chooseAuditPlan(nuclei, layout, rng);
      auditOrder = plan.auditOrder;
      selectIds = plan.selectIds;
      queue = [];
      flights = [];
      patches = null;
      auditIndex = -1;
      phase = "enter";
      phaseStartedAt = 0;
      storyTime = 0;
      nonPriorityDim = 1;
      studyReveal = 0;
      finishedOnce = false;
      camera = { scale: 1, velocity: 0, target: 1, x: 0, y: 0, tx: 0, ty: 0, vx: 0, vy: 0 };
      frame.visible = false;
      frame.activeAccent = 0;

      const enter = paced(CONFIG.timing.sceneEnterMs);
      for (let i = 0; i < nuclei.length; i += 1) {
        nuclei[i].appearAt = (i / Math.max(1, nuclei.length - 1)) * enter * 0.55;
        nuclei[i].selected = false;
        nuclei[i].marked = false;
        nuclei[i].markProgress = 0;
        nuclei[i].markStartedAt = -1;
        nuclei[i].dim = 1;
      }
    }

    function startFrameTravel(toNucleus, now) {
      const pos = worldOf(toNucleus);
      const size = Math.max(toNucleus.rx, toNucleus.ry) * 2.35;
      const duration = paced(pickRange(rng, CONFIG.timing.frameTravelMs));
      frame.fromX = frame.visible ? frame.x : pos.x;
      frame.fromY = frame.visible ? frame.y : pos.y;
      frame.fromSize = frame.visible ? frame.size : size * 0.85;
      frame.toX = pos.x;
      frame.toY = pos.y;
      frame.toSize = size;
      // Soft bezier controls offset perpendicular to travel.
      const dx = frame.toX - frame.fromX;
      const dy = frame.toY - frame.fromY;
      const len = Math.hypot(dx, dy) || 1;
      const px = (-dy / len) * (28 + rng() * 36);
      const py = (dx / len) * (28 + rng() * 36);
      const side = rng() < 0.5 ? 1 : -1;
      frame.c1x = frame.fromX + dx * 0.28 + px * side;
      frame.c1y = frame.fromY + dy * 0.28 + py * side;
      frame.c2x = frame.fromX + dx * 0.72 - px * side * 0.6;
      frame.c2y = frame.fromY + dy * 0.72 - py * side * 0.6;
      frame.startedAt = now;
      frame.duration = duration;
      frame.visible = true;
      frame.settleUntil = 0;
      frame.inspectUntil = 0;
      frame.activeAccent = 0;
    }

    function beginAudit(now) {
      auditIndex = 0;
      phase = "travel";
      phaseStartedAt = now;
      startFrameTravel(nuclei[auditOrder[0]], now);
    }

    function beginMarkAndFlight(nucleus, now) {
      nucleus.selected = true;
      nucleus.marked = true;
      nucleus.markStartedAt = now;
      nucleus.markProgress = 0;
      phase = "mark";
      phaseStartedAt = now;
    }

    function launchCopy(nucleus, now) {
      const from = worldOf(nucleus);
      const slot = queue.length;
      const dest = queueSlotPos(slot);
      const duration = paced(pickRange(rng, CONFIG.timing.copyFlightMs));
      const dx = dest.x - from.x;
      const dy = dest.y - from.y;
      const len = Math.hypot(dx, dy) || 1;
      const px = (-dy / len) * (40 + rng() * 50);
      const py = (dx / len) * (40 + rng() * 50);
      flights.push({
        nucleusId: nucleus.id,
        fromX: from.x,
        fromY: from.y,
        toX: dest.x,
        toY: dest.y,
        c1x: from.x + dx * 0.3 + px,
        c1y: from.y + dy * 0.3 + py,
        c2x: from.x + dx * 0.75 - px * 0.35,
        c2y: from.y + dy * 0.75 - py * 0.35,
        fromScale: 1,
        toScale: dest.scale,
        fromRot: nucleus.rot,
        toRot: nucleus.rot * 0.25,
        startedAt: now,
        duration,
        done: false,
        slot,
        path: nucleus.path,
        rx: nucleus.rx,
        ry: nucleus.ry,
        annPad: nucleus.annPad,
      });
      phase = "flight";
      phaseStartedAt = now;
    }

    function advanceAfterInspect(now) {
      const id = auditOrder[auditIndex];
      const nucleus = nuclei[id];
      if (selectIds.has(id) && !nucleus.selected && queue.length < layout.queueSlots) {
        beginMarkAndFlight(nucleus, now);
        return;
      }
      goNextAuditOrHold(now);
    }

    function goNextAuditOrHold(now) {
      auditIndex += 1;
      if (auditIndex >= auditOrder.length || queue.length >= layout.queueSlots) {
        // If queue not full but audit done, force remaining selections quietly.
        if (queue.length < layout.queueSlots) {
          for (const id of selectIds) {
            if (queue.length >= layout.queueSlots) break;
            const n = nuclei[id];
            if (n.selected) continue;
            beginMarkAndFlight(n, now);
            return;
          }
        }
        phase = "queue_hold";
        phaseStartedAt = now;
        frame.activeAccent = 0;
        return;
      }
      phase = "travel";
      phaseStartedAt = now;
      startFrameTravel(nuclei[auditOrder[auditIndex]], now);
    }

    function beginZoomSequence(now) {
      phase = "zoom_patch";
      phaseStartedAt = now;
      camera.target = 0.78;
      camera.tx = isMobile ? 0 : width * 0.02;
      camera.ty = isMobile ? height * -0.03 : 0;
      nonPriorityDim = 1;
      const baseX = width * layout.fieldCx;
      const baseY = height * layout.fieldCy;
      const offsets = [
        { x: 0, y: 0, live: true },
        { x: -0.22, y: -0.2, live: false },
        { x: 0.23, y: -0.08, live: false },
        { x: 0.18, y: 0.2, live: false },
      ];
      patches = offsets.map((d, i) => {
        const localRng = mulberry32(CONFIG.seed ^ (0xb10b + i * 97));
        const dots = [];
        if (!d.live) {
          const count = 9 + Math.floor(localRng() * 5);
          for (let k = 0; k < count; k += 1) {
            const ang = localRng() * Math.PI * 2;
            const rad = Math.sqrt(localRng()) * 0.42;
            dots.push({
              x: Math.cos(ang) * rad,
              y: Math.sin(ang) * rad,
              rx: 3.2 + localRng() * 2.4,
              ry: 2.6 + localRng() * 2.2,
              rot: (localRng() - 0.5) * 0.8,
            });
          }
        }
        return {
          x: baseX + d.x * fieldScale,
          y: baseY + d.y * fieldScale,
          live: d.live,
          appear: d.live ? 1 : 0,
          dots,
        };
      });
    }

    function updateCamera(dtMs) {
      const dtCap = Math.min(dtMs, CONFIG.timing.maxFrameDeltaMs);
      const substeps = Math.max(1, Math.ceil(dtCap / CONFIG.timing.springSubstepMs));
      const step = dtCap / substeps / 1000;
      const { stiffness, damping } = CONFIG.camera;
      for (let i = 0; i < substeps; i += 1) {
        const ax = (camera.target - camera.scale) * stiffness - camera.velocity * damping;
        camera.velocity += ax * step;
        camera.scale += camera.velocity * step;

        const axx = (camera.tx - camera.x) * stiffness - camera.vx * damping;
        const ayy = (camera.ty - camera.y) * stiffness - camera.vy * damping;
        camera.vx += axx * step;
        camera.vy += ayy * step;
        camera.x += camera.vx * step;
        camera.y += camera.vy * step;
      }
    }

    function cameraSettled() {
      return (
        Math.abs(camera.target - camera.scale) < CONFIG.camera.settleEpsilon &&
        Math.abs(camera.velocity) < CONFIG.camera.settleEpsilon &&
        Math.abs(camera.tx - camera.x) < 0.4 &&
        Math.abs(camera.ty - camera.y) < 0.4
      );
    }

    function updateContinuousMotion(now, delta) {
      // Frame position along bezier.
      if (frame.visible && (phase === "travel" || phase === "settle" || phase === "inspect" || phase === "mark" || phase === "flight")) {
        if (phase === "travel") {
          const raw = clamp((now - frame.startedAt) / frame.duration, 0, 1);
          const e = easeInOutCubic(raw);
          frame.x = cubicBezier(frame.fromX, frame.c1x, frame.c2x, frame.toX, e);
          frame.y = cubicBezier(frame.fromY, frame.c1y, frame.c2y, frame.toY, e);
          frame.size = lerp(frame.fromSize, frame.toSize, e);
          frame.activeAccent = easeOutCubic(raw);
          if (raw >= 1) {
            phase = "settle";
            phaseStartedAt = now;
            frame.settleUntil = now + paced(pickRange(rng, CONFIG.timing.frameSettleMs));
            frame.x = frame.toX;
            frame.y = frame.toY;
            frame.size = frame.toSize;
          }
        } else {
          frame.x = frame.toX;
          frame.y = frame.toY;
          frame.size = frame.toSize;
          frame.activeAccent = phase === "inspect" || phase === "mark" ? 1 : lerp(frame.activeAccent, 0.35, 0.08);
        }
      }

      // Mark draw progress.
      for (let i = 0; i < nuclei.length; i += 1) {
        const n = nuclei[i];
        if (n.markStartedAt >= 0) {
          n.markProgress = clamp((now - n.markStartedAt) / paced(CONFIG.timing.markDrawMs), 0, 1);
        }
      }

      // Flights.
      for (let i = 0; i < flights.length; i += 1) {
        const f = flights[i];
        if (f.done) continue;
        const raw = clamp((now - f.startedAt) / f.duration, 0, 1);
        const e = easeInOutQuint(raw);
        f.x = cubicBezier(f.fromX, f.c1x, f.c2x, f.toX, e);
        f.y = cubicBezier(f.fromY, f.c1y, f.c2y, f.toY, e);
        f.scale = lerp(f.fromScale, f.toScale, e);
        f.rot = lerp(f.fromRot, f.toRot, e);
        f.opacity = lerp(0.55, 1, easeOutCubic(Math.min(1, raw * 1.4)));
        if (raw >= 1) {
          f.done = true;
          f.x = f.toX;
          f.y = f.toY;
          f.scale = f.toScale;
          queue.push({
            nucleusId: f.nucleusId,
            x: f.toX,
            y: f.toY,
            scale: f.toScale,
            appearAt: now,
            path: f.path,
            rx: f.rx,
            ry: f.ry,
            rot: f.toRot,
            annPad: f.annPad,
          });
        }
      }

      // Dim non-priority during hold/zoom.
      if (phase === "queue_hold" || phase === "zoom_patch" || phase === "zoom_study" || phase === "final" || phase === "done") {
        const targetDim = 0.28;
        nonPriorityDim = lerp(nonPriorityDim, targetDim, 1 - Math.exp(-delta * 0.004));
        for (let i = 0; i < nuclei.length; i += 1) {
          nuclei[i].dim = nuclei[i].selected ? 1 : nonPriorityDim;
        }
      }

      if (patches) {
        const t = clamp((now - phaseStartedAt) / paced(900), 0, 1);
        for (let i = 0; i < patches.length; i += 1) {
          if (patches[i].live) patches[i].appear = 1;
          else patches[i].appear = easeOutCubic(clamp(t - i * 0.08, 0, 1));
        }
      }

      if (phase === "zoom_patch" || phase === "zoom_study" || phase === "final" || phase === "done") {
        updateCamera(delta);
      }
    }

    function updateSimulation(now) {
      // Discrete phase transitions only - no visual jumps here.
      if (phase === "enter") {
        if (now - phaseStartedAt >= paced(CONFIG.timing.sceneEnterMs)) {
          beginAudit(now);
        }
        return;
      }

      if (phase === "settle") {
        if (now >= frame.settleUntil) {
          phase = "inspect";
          phaseStartedAt = now;
          frame.inspectUntil = now + paced(pickRange(rng, CONFIG.timing.inspectMs));
        }
        return;
      }

      if (phase === "inspect") {
        if (now >= frame.inspectUntil) {
          advanceAfterInspect(now);
        }
        return;
      }

      if (phase === "mark") {
        if (now - phaseStartedAt >= paced(CONFIG.timing.markDrawMs)) {
          const id = auditOrder[auditIndex] ?? [...selectIds][queue.length];
          // Prefer currently marked nucleus not yet queued.
          let nucleus = null;
          for (let i = 0; i < nuclei.length; i += 1) {
            if (nuclei[i].selected && !queue.some((q) => q.nucleusId === nuclei[i].id) && !flights.some((f) => !f.done && f.nucleusId === nuclei[i].id)) {
              nucleus = nuclei[i];
              break;
            }
          }
          if (!nucleus) nucleus = nuclei[id];
          launchCopy(nucleus, now);
        }
        return;
      }

      if (phase === "flight") {
        const pending = flights.some((f) => !f.done);
        if (!pending) {
          goNextAuditOrHold(now);
        }
        return;
      }

      if (phase === "queue_hold") {
        if (now - phaseStartedAt >= paced(CONFIG.timing.queueHoldMs)) {
          beginZoomSequence(now);
        }
        return;
      }

      if (phase === "zoom_patch") {
        const elapsed = now - phaseStartedAt;
        if (elapsed >= paced(CONFIG.timing.zoomToPatchMs) || (elapsed > 900 && cameraSettled())) {
          phase = "zoom_study";
          phaseStartedAt = now;
          camera.target = 0.68;
          camera.tx = 0;
          camera.ty = isMobile ? height * -0.04 : 0;
          studyReveal = 1;
        }
        return;
      }

      if (phase === "zoom_study") {
        const elapsed = now - phaseStartedAt;
        if (elapsed >= paced(CONFIG.timing.zoomToStudyMs) || (elapsed > 900 && cameraSettled())) {
          phase = "final";
          phaseStartedAt = now;
          camera.target = 0.68;
        }
        return;
      }

      if (phase === "final") {
        if (now - phaseStartedAt >= paced(CONFIG.timing.finalSettleMs)) {
          phase = "done";
          finishedOnce = true;
        }
      }
    }

    function appearAmount(bornAt, now) {
      const t = clamp((now - bornAt) / paced(CONFIG.timing.appearMs), 0, 1);
      return easeOutCubic(t);
    }

    function drawReviewCorners(g, x, y, size, progress, color, alpha) {
      const p = easeOutCubic(clamp(progress, 0, 1));
      const half = size * 0.5;
      const inset = size * 0.18;
      const len = size * 0.22 * p;
      g.strokeStyle = rgba(color, alpha);
      g.lineWidth = Math.max(1.2, size * 0.045);
      g.lineCap = "round";
      g.beginPath();
      // TL
      g.moveTo(x - half + inset, y - half + inset + len);
      g.lineTo(x - half + inset, y - half + inset);
      g.lineTo(x - half + inset + len, y - half + inset);
      // TR
      g.moveTo(x + half - inset - len, y - half + inset);
      g.lineTo(x + half - inset, y - half + inset);
      g.lineTo(x + half - inset, y - half + inset + len);
      // BR
      g.moveTo(x + half - inset, y + half - inset - len);
      g.lineTo(x + half - inset, y + half - inset);
      g.lineTo(x + half - inset - len, y + half - inset);
      // BL
      g.moveTo(x - half + inset + len, y + half - inset);
      g.lineTo(x - half + inset, y + half - inset);
      g.lineTo(x - half + inset, y + half - inset - len);
      g.stroke();
    }

    function drawOpenFrame(g, x, y, size, accent) {
      const half = size * 0.5;
      const len = size * 0.22;
      const inset = size * 0.02;
      const col = mixAccent(accent);
      g.strokeStyle = rgba(col, 0.55 + accent * 0.35);
      g.lineWidth = Math.max(1.25, size * 0.035);
      g.lineCap = "round";
      g.beginPath();
      g.moveTo(x - half + inset, y - half + inset + len);
      g.lineTo(x - half + inset, y - half + inset);
      g.lineTo(x - half + inset + len, y - half + inset);
      g.moveTo(x + half - inset - len, y - half + inset);
      g.lineTo(x + half - inset, y - half + inset);
      g.lineTo(x + half - inset, y - half + inset + len);
      g.moveTo(x + half - inset, y + half - inset - len);
      g.lineTo(x + half - inset, y + half - inset);
      g.lineTo(x + half - inset - len, y + half - inset);
      g.moveTo(x - half + inset + len, y + half - inset);
      g.lineTo(x - half + inset, y + half - inset);
      g.lineTo(x - half + inset, y + half - inset - len);
      g.stroke();
    }

    function mixAccent(t) {
      const a = CONFIG.colors.selected;
      const b = CONFIG.colors.active;
      return [
        Math.round(lerp(a[0], b[0], t)),
        Math.round(lerp(a[1], b[1], t)),
        Math.round(lerp(a[2], b[2], t)),
      ];
    }

    function drawNucleusShape(g, n, x, y, scale, opacity, opts) {
      const selected = opts.selected;
      const markP = opts.markProgress || 0;
      g.save();
      g.translate(x, y);
      g.rotate(n.rot);
      g.scale(scale, scale);
      g.globalAlpha = opacity;

      // Dark fill body
      g.fillStyle = rgba(CONFIG.colors.fill, opts.queueItem ? 0.98 : 0.92);
      g.fill(n.path);

      // Nucleus body
      g.fillStyle = rgba(CONFIG.colors.nucleusBody, opts.queueItem ? 0.92 : 0.75);
      g.fill(n.path);

      // Source annotation outline (slightly padded via stroke width)
      g.strokeStyle = rgba(
        selected ? CONFIG.colors.selected : CONFIG.colors.annotation,
        selected ? 0.9 : 0.55
      );
      g.lineWidth = (selected ? 1.7 : 1.15) + n.annPad * 0.15;
      g.stroke(n.path);

      g.restore();

      if (selected || markP > 0) {
        const size = Math.max(n.rx, n.ry) * 2.4 * scale;
        drawReviewCorners(
          g,
          x,
          y,
          size,
          markP > 0 ? markP : 1,
          CONFIG.colors.selected,
          opacity * (0.55 + (selected ? 0.35 : 0))
        );
      }
    }

    function drawField(now) {
      for (let i = 0; i < nuclei.length; i += 1) {
        const n = nuclei[i];
        const born = appearAmount(n.appearAt, now);
        if (born <= 0.01) continue;
        const pos = worldOf(n);
        const scale = lerp(0.86, 1, born);
        const opacity = born * n.dim * (isMobile ? 0.82 : 1);
        drawNucleusShape(ctx, n, pos.x, pos.y, scale, opacity, {
          selected: n.selected,
          markProgress: n.markProgress,
        });
      }
    }

    function drawFlights() {
      for (let i = 0; i < flights.length; i += 1) {
        const f = flights[i];
        if (f.done) continue;
        const n = {
          path: f.path,
          rx: f.rx,
          ry: f.ry,
          rot: f.rot,
          annPad: f.annPad,
        };
        drawNucleusShape(ctx, n, f.x, f.y, f.scale, f.opacity, {
          selected: true,
          markProgress: 1,
        });
      }
    }

    function drawQueue(now) {
      for (let i = 0; i < queue.length; i += 1) {
        const q = queue[i];
        const born = appearAmount(q.appearAt, now);
        const n = {
          path: q.path,
          rx: q.rx,
          ry: q.ry,
          rot: q.rot,
          annPad: q.annPad,
        };
        const emphasis = i === 0 ? 1 : 0.88;
        // Queue miniatures stay organic: draw a bit larger and brighter than field copies.
        drawNucleusShape(
          ctx,
          n,
          q.x,
          q.y,
          q.scale * 1.15 * lerp(0.86, 1, born),
          Math.min(1, born * emphasis + 0.08),
          {
            selected: true,
            markProgress: 1,
            queueItem: true,
          }
        );
      }
    }

    function drawStudyPatches() {
      if (!patches || studyReveal <= 0) return;
      for (let i = 0; i < patches.length; i += 1) {
        const p = patches[i];
        if (p.live) continue;
        const a = p.appear * (isMobile ? 0.55 : 0.75);
        if (a <= 0.01) continue;
        const span = fieldScale * 0.16;
        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.globalAlpha = a;
        for (let k = 0; k < p.dots.length; k += 1) {
          const d = p.dots[k];
          ctx.save();
          ctx.translate(d.x * span * 2.2, d.y * span * 2.2);
          ctx.rotate(d.rot);
          ctx.fillStyle = rgba(CONFIG.colors.nucleusBody, 0.7);
          ctx.strokeStyle = rgba(CONFIG.colors.annotationSoft, 0.55);
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.ellipse(0, 0, d.rx, d.ry, 0, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
          ctx.restore();
        }
        ctx.restore();
      }
    }

    function render(now) {
      ctx.clearRect(0, 0, width, height);
      ctx.save();
      const ox = width * 0.5 + camera.x;
      const oy = height * 0.5 + camera.y;
      ctx.translate(ox, oy);
      ctx.scale(camera.scale, camera.scale);
      ctx.translate(-width * 0.5, -height * 0.5);

      drawStudyPatches();
      drawField(now);
      drawFlights();
      drawQueue(now);

      if (frame.visible && phase !== "done" && phase !== "final" && phase !== "zoom_patch" && phase !== "zoom_study" && phase !== "queue_hold") {
        drawOpenFrame(ctx, frame.x, frame.y, frame.size, frame.activeAccent);
      } else if (frame.visible && phase === "queue_hold") {
        drawOpenFrame(ctx, frame.x, frame.y, frame.size, 0.2);
      }

      ctx.restore();
    }

    function renderStaticFinal() {
      rebuildScene();
      // Instantly resolve to final storytelling frame.
      for (const id of selectIds) {
        const n = nuclei[id];
        n.selected = true;
        n.marked = true;
        n.markProgress = 1;
        n.markStartedAt = 0;
      }
      for (let i = 0; i < nuclei.length; i += 1) {
        nuclei[i].dim = nuclei[i].selected ? 1 : 0.3;
        nuclei[i].appearAt = -1000;
      }
      let slot = 0;
      for (const id of selectIds) {
        const n = nuclei[id];
        const dest = queueSlotPos(slot);
        queue.push({
          nucleusId: id,
          x: dest.x,
          y: dest.y,
          scale: dest.scale,
          appearAt: -1000,
          path: n.path,
          rx: n.rx,
          ry: n.ry,
          rot: n.rot * 0.25,
          annPad: n.annPad,
        });
        slot += 1;
      }
      beginZoomSequence(0);
      camera.scale = 0.68;
      camera.target = 0.68;
      camera.velocity = 0;
      camera.x = 0;
      camera.y = isMobile ? height * -0.04 : 0;
      camera.tx = camera.x;
      camera.ty = camera.y;
      studyReveal = 1;
      for (const p of patches) p.appear = 1;
      phase = "done";
      finishedOnce = true;
      storyTime = 30000;
      render(storyTime);
    }

    function frameLoop(now) {
      if (!running) return;
      if (!previousFrameTime) previousFrameTime = now;
      const rawDelta = now - previousFrameTime;
      const delta = Math.min(rawDelta, CONFIG.timing.maxFrameDeltaMs);
      previousFrameTime = now;
      accumulator += rawDelta;

      const step = CONFIG.timing.fixedStepMs;
      // Advance story clock with capped real time (not fixed-step only).
      storyTime += delta;

      while (accumulator >= step) {
        updateSimulation(storyTime);
        accumulator -= step;
      }

      updateContinuousMotion(storyTime, delta);
      render(storyTime);

      if (phase === "done") {
        stopLoop(false);
        return;
      }
      raf = window.requestAnimationFrame(frameLoop);
    }

    function startLoop() {
      if (reduced || running) return;
      if (finishedOnce && phase === "done") {
        render(storyTime);
        return;
      }
      if (!visible || !pageVisible) return;
      running = true;
      previousFrameTime = 0;
      accumulator = 0;
      raf = window.requestAnimationFrame(frameLoop);
    }

    function stopLoop(resetClock) {
      running = false;
      if (raf) {
        window.cancelAnimationFrame(raf);
        raf = 0;
      }
      if (resetClock) {
        previousFrameTime = 0;
        accumulator = 0;
      }
    }

    function onVisibility() {
      pageVisible = !document.hidden;
      if (pageVisible && visible) {
        previousFrameTime = 0;
        accumulator = 0;
        startLoop();
      } else stopLoop(true);
    }

    function onIntersect(show) {
      if (show) {
        if (finishedOnce && phase === "done") {
          rebuildScene();
          phaseStartedAt = 0;
          storyTime = 0;
        }
        visible = true;
        if (reduced) {
          renderStaticFinal();
          return;
        }
        if (phase === "enter" && storyTime === 0) {
          phaseStartedAt = 0;
        }
        startLoop();
      } else {
        visible = false;
        stopLoop(true);
      }
    }

    measure();
    rebuildScene();
    if (reduced) renderStaticFinal();
    else render(0);

    const ro = new ResizeObserver(() => {
      const wasMobile = isMobile;
      measure();
      if (wasMobile !== isMobile) {
        const keepDone = finishedOnce && phase === "done";
        rebuildScene();
        if (keepDone || reduced) renderStaticFinal();
        else {
          phaseStartedAt = storyTime;
          render(storyTime);
        }
      } else {
        // Recompute queue positions for current slot contents.
        for (let i = 0; i < queue.length; i += 1) {
          const dest = queueSlotPos(i);
          queue[i].x = dest.x;
          queue[i].y = dest.y;
          queue[i].scale = dest.scale;
        }
        render(storyTime);
      }
    });
    ro.observe(canvas.parentElement || canvas);

    const io = new IntersectionObserver(
      (entries) => {
        const entry = entries[0];
        onIntersect(Boolean(entry && entry.intersectionRatio >= 0.1));
      },
      { threshold: [0, 0.1, 0.25] }
    );
    io.observe(canvas.parentElement || canvas);

    document.addEventListener("visibilitychange", onVisibility);

    return function destroy() {
      stopLoop(true);
      ro.disconnect();
      io.disconnect();
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }

  function boot() {
    const canvas = document.querySelector("canvas.hero-review-canvas");
    if (!canvas) return;
    const destroy = createController(canvas);
    window.addEventListener(
      "pagehide",
      () => {
        if (typeof destroy === "function") destroy();
      },
      { once: true }
    );
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    boot();
  }
})();
