(() => {
  "use strict";

  /**
   * Second-Look Review Field.
   *
   * This is a decorative Canvas 2D story. It never changes an annotation:
   * marked source sprites stay in the field while identical visual copies move
   * into a short expert-review queue.
   */

  const STATES = Object.freeze({
    PRELOAD: "PRELOAD",
    INTRO: "INTRO",
    SCAN: "SCAN",
    SELECT_AND_CLONE: "SELECT_AND_CLONE",
    QUEUE_COMPLETE: "QUEUE_COMPLETE",
    ZOOM_TO_PATCH: "ZOOM_TO_PATCH",
    ZOOM_TO_STUDY: "ZOOM_TO_STUDY",
    SETTLED: "SETTLED",
    RETURN_TO_FIELD: "RETURN_TO_FIELD",
  });

  const ASSET_ROOT = "assets/hero/nuclei/";
  const ASSET_DEFINITIONS = Object.freeze([
    { filename: "nucleus-compact.png", fallbackAspect: 1 },
    { filename: "nucleus-elongated.png", fallbackAspect: 1.5 },
    { filename: "nucleus-kidney.png", fallbackAspect: 1 },
    { filename: "nucleus-bilobed.png", fallbackAspect: 1 },
    { filename: "nucleus-irregular.png", fallbackAspect: 1 },
    { filename: "nucleus-flattened.png", fallbackAspect: 1.5 },
  ]);

  const CONFIG = Object.freeze({
    seed: 0x4a11ca27,
    breakpoint: 720,
    fixedStepSeconds: 1 / 60,
    renderIntervalSeconds: 1 / 60,
    maxFrameDeltaSeconds: 0.05,
    cameraSubstepSeconds: 0.012,
    timings: Object.freeze({
      intro: 0.92,
      frameTravelMin: 0.82,
      frameTravelMax: 1,
      frameSettleMin: 0.15,
      frameSettleMax: 0.21,
      inspectMin: 0.44,
      inspectMax: 0.58,
      mark: 0.34,
      copyFlightMin: 0.8,
      copyFlightMax: 1,
      queueComplete: 0.68,
      zoomToPatch: 2,
      zoomToStudy: 2.45,
      logoDisplay: 1.8,
      returnSpinDive: 1.35,
      returnFadeToBlack: 0.22,
      returnBlackHold: 0.195,
      returnRevealField: 0.46,
      returnFieldHold: 0.14,
    }),
    fieldReveal: Object.freeze({
      groupCount: 8,
      staggerSpan: 0.455,
      blackReleaseEnd: 0.24,
    }),
    morph: Object.freeze({
      cameraEnd: 0.72,
      siblingStart: 0.02,
      siblingEnd: 0.62,
      rotateStart: 0.26,
      rotateEnd: 0.9,
      colorStart: 0.52,
      colorEnd: 1,
    }),
    dive: Object.freeze({ endScale: 120, fadeScale: 150 }),
    renderProfiles: Object.freeze({
      constrained: Object.freeze({ dprCap: 1.25, pixelBudget: 2200000, cacheSize: 384 }),
      balanced: Object.freeze({ dprCap: 1.5, pixelBudget: 3200000, cacheSize: 448 }),
      high: Object.freeze({ dprCap: 1.75, pixelBudget: 5000000, cacheSize: 512 }),
    }),
    camera: Object.freeze({
      stiffness: 19,
      damping: 9.2,
      positionEpsilon: 0.25,
      scaleEpsilon: 0.0008,
      velocityEpsilon: 0.002,
    }),
    desktop: Object.freeze({
      nucleusCount: 58,
      auditCount: 4,
      selectCount: 4,
      selectedScanPositions: Object.freeze([0, 1, 2, 3]),
      patchScale: 0.82,
      patchScreenX: 0.73,
      patchScreenY: 0.52,
      studyScreenX: 0.74,
      studyScreenY: 0.55,
      logoScreenX: 0.74,
      logoScreenY: 0.55,
      studyPatchFraction: 0.068,
      studyPatchMin: 50,
      studyPatchMax: 104,
      logoPatchFraction: 0.043,
      logoPatchMin: 34,
      logoPatchMax: 66,
      baseNucleusHeight: 0.092,
      textClearance: 30,
      mobileAlpha: 1,
      queueHorizontal: false,
    }),
    mobile: Object.freeze({
      nucleusCount: 30,
      auditCount: 3,
      selectCount: 3,
      selectedScanPositions: Object.freeze([0, 1, 2]),
      patchScale: 0.92,
      patchScreenX: 0.5,
      patchScreenY: 0.68,
      studyScreenX: 0.5,
      studyScreenY: 0.74,
      logoScreenX: 0.5,
      logoScreenY: 0.74,
      studyPatchFraction: 0.13,
      studyPatchMin: 42,
      studyPatchMax: 52,
      logoPatchFraction: 0.078,
      logoPatchMin: 26,
      logoPatchMax: 34,
      baseNucleusHeight: 0.112,
      textClearance: 22,
      mobileAlpha: 0.94,
      queueHorizontal: true,
    }),
  });

  const COLORS = Object.freeze({
    canvas: "#010102",
    surface: "#0f1011",
    body: "#18191a",
    hairline: "#23252a",
    hairlineStrong: "#34343a",
    annotation: "#62666d",
    accent: "#5e6ad2",
    accentBright: "#828fff",
  });

  const REGIONS = Object.freeze([
    Object.freeze({ x: -0.21, y: -0.2, rx: 0.18, ry: 0.16 }),
    Object.freeze({ x: 0.13, y: -0.18, rx: 0.17, ry: 0.15 }),
    Object.freeze({ x: -0.18, y: 0.18, rx: 0.17, ry: 0.16 }),
    Object.freeze({ x: 0.15, y: 0.19, rx: 0.16, ry: 0.15 }),
  ]);

  function clamp(value, minimum, maximum) {
    return value < minimum ? minimum : value > maximum ? maximum : value;
  }

  function detectRenderQuality() {
    const hardwareThreads = Number(navigator.hardwareConcurrency) || 8;
    const deviceMemory = Number(navigator.deviceMemory) || 8;
    const connection = navigator.connection;
    const saveData = Boolean(connection && connection.saveData);

    if (saveData || hardwareThreads <= 4 || deviceMemory <= 4) return "constrained";
    if (hardwareThreads >= 8 && deviceMemory >= 8) return "high";
    return "balanced";
  }

  function lerp(from, to, progress) {
    return from + (to - from) * progress;
  }

  function easeInOutCubic(progress) {
    return progress < 0.5
      ? 4 * progress * progress * progress
      : 1 - Math.pow(-2 * progress + 2, 3) / 2;
  }

  function easeOutCubic(progress) {
    const inverse = 1 - progress;
    return 1 - inverse * inverse * inverse;
  }

  function easeInCubic(progress) {
    return progress * progress * progress;
  }

  function easeOutBack(progress) {
    const overshoot = 1.70158;
    const shifted = progress - 1;
    return 1 + (overshoot + 1) * shifted * shifted * shifted + overshoot * shifted * shifted;
  }

  function easeInOutQuint(progress) {
    return progress < 0.5
      ? 16 * progress * progress * progress * progress * progress
      : 1 - Math.pow(-2 * progress + 2, 5) / 2;
  }

  function cubicBezier(from, controlOne, controlTwo, to, progress) {
    const inverse = 1 - progress;
    const inverseSquared = inverse * inverse;
    const progressSquared = progress * progress;
    return (
      inverseSquared * inverse * from +
      3 * inverseSquared * progress * controlOne +
      3 * inverse * progressSquared * controlTwo +
      progressSquared * progress * to
    );
  }

  function createSeededRandom(seed) {
    let state = seed >>> 0;
    return () => {
      state += 0x6d2b79f5;
      let value = Math.imul(state ^ (state >>> 15), 1 | state);
      value ^= value + Math.imul(value ^ (value >>> 7), 61 | value);
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
  }

  function randomRange(random, minimum, maximum) {
    return lerp(minimum, maximum, random());
  }

  function shuffleInPlace(values, random) {
    for (let index = values.length - 1; index > 0; index -= 1) {
      const swapIndex = Math.floor(random() * (index + 1));
      const current = values[index];
      values[index] = values[swapIndex];
      values[swapIndex] = current;
    }
  }

  function makeFallbackPath() {
    const path = new Path2D();
    path.moveTo(-0.48, -0.05);
    path.bezierCurveTo(-0.46, -0.35, -0.2, -0.5, 0.08, -0.43);
    path.bezierCurveTo(0.42, -0.36, 0.5, -0.1, 0.42, 0.16);
    path.bezierCurveTo(0.34, 0.43, 0.04, 0.5, -0.22, 0.4);
    path.bezierCurveTo(-0.45, 0.3, -0.53, 0.12, -0.48, -0.05);
    path.closePath();
    return path;
  }

  function makePatchPath() {
    const path = new Path2D();
    const half = 0.5;
    const radius = 0.045;
    path.moveTo(-half + radius, -half);
    path.lineTo(half - radius, -half);
    path.quadraticCurveTo(half, -half, half, -half + radius);
    path.lineTo(half, half - radius);
    path.quadraticCurveTo(half, half, half - radius, half);
    path.lineTo(-half + radius, half);
    path.quadraticCurveTo(-half, half, -half, half - radius);
    path.lineTo(-half, -half + radius);
    path.quadraticCurveTo(-half, -half, -half + radius, -half);
    path.closePath();
    return path;
  }

  function makeLogoTilePath() {
    const path = new Path2D();
    const half = 0.5;
    const radius = 0.22;
    path.moveTo(-half + radius, -half);
    path.lineTo(half - radius, -half);
    path.quadraticCurveTo(half, -half, half, -half + radius);
    path.lineTo(half, half - radius);
    path.quadraticCurveTo(half, half, half - radius, half);
    path.lineTo(-half + radius, half);
    path.quadraticCurveTo(-half, half, -half, half - radius);
    path.lineTo(-half, -half + radius);
    path.quadraticCurveTo(-half, -half, -half + radius, -half);
    path.closePath();
    return path;
  }

  const FALLBACK_PATH = makeFallbackPath();
  const PATCH_PATH = makePatchPath();
  const LOGO_TILE_PATH = makeLogoTilePath();

  async function loadSprite(definition) {
    const image = new Image();
    image.decoding = "async";
    image.src = new URL(ASSET_ROOT + definition.filename, document.baseURI).href;

    try {
      await new Promise((resolve, reject) => {
        if (image.complete && image.naturalWidth > 0) {
          resolve();
          return;
        }
        image.addEventListener("load", resolve, { once: true });
        image.addEventListener("error", reject, { once: true });
      });
      if (typeof image.decode === "function") {
        try {
          await image.decode();
        } catch (_decodeError) {
          // A completed image remains drawable even when decode() rejects late.
        }
      }

      const naturalWidth = image.naturalWidth;
      const naturalHeight = image.naturalHeight;
      if (!(naturalWidth > 0 && naturalHeight > 0)) throw new Error("empty image");

      const maximumDimension = 384;
      const resizeScale = Math.min(1, maximumDimension / Math.max(naturalWidth, naturalHeight));
      const cacheWidth = Math.max(1, Math.round(naturalWidth * resizeScale));
      const cacheHeight = Math.max(1, Math.round(naturalHeight * resizeScale));
      const cache = document.createElement("canvas");
      cache.width = cacheWidth;
      cache.height = cacheHeight;
      const cacheContext = cache.getContext("2d", { alpha: true });
      if (!cacheContext) throw new Error("sprite cache unavailable");
      cacheContext.filter = "brightness(1.35) contrast(1.08) saturate(1.08)";
      cacheContext.drawImage(image, 0, 0, cacheWidth, cacheHeight);
      cacheContext.filter = "none";

      let source = cache;
      let closeable = false;
      if (typeof window.createImageBitmap === "function") {
        try {
          source = await window.createImageBitmap(cache);
          closeable = true;
        } catch (_bitmapError) {
          source = cache;
        }
      }

      return {
        filename: definition.filename,
        source,
        closeable,
        ready: true,
        aspect: naturalWidth / naturalHeight,
      };
    } catch (_loadError) {
      return {
        filename: definition.filename,
        source: null,
        closeable: false,
        ready: false,
        aspect: definition.fallbackAspect,
      };
    }
  }

  function createController(canvas) {
    const context = canvas.getContext("2d", { alpha: true });
    if (!context) return null;

    const hero = canvas.closest(".hero");
    const heroCopy = hero ? hero.querySelector(".hero-copy") : null;
    const motionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    const qualityTier = detectRenderQuality();
    const renderProfile = CONFIG.renderProfiles[qualityTier];

    let reducedMotion = motionQuery.matches;
    let destroyed = false;
    let assetsLoaded = false;
    let sprites = ASSET_DEFINITIONS.map((definition) => ({
      filename: definition.filename,
      source: null,
      closeable: false,
      ready: false,
      aspect: definition.fallbackAspect,
    }));
    let patchCaches = [];

    let width = 1;
    let height = 1;
    let dpr = 1;
    let isMobile = false;
    let layout = CONFIG.desktop;
    let patchSize = 1;
    let gridSpacing = 1;
    let textSafeLeft = -1;
    let textSafeTop = -1;
    let textSafeRight = -1;
    let textSafeBottom = -1;

    const patchCenters = new Float64Array(8);
    const initialCamera = { scale: 1, x: 0, y: 0 };
    const patchCamera = { scale: 0.76, x: 0, y: 0 };
    const studyCamera = { scale: 0.22, x: 0, y: 0 };
    const finalCamera = { scale: 0.4, x: 0, y: 0 };
    const zoomStartCamera = { scale: 1, x: 0, y: 0 };

    const camera = {
      scale: 1,
      previousScale: 1,
      velocityScale: 0,
      goalScale: 1,
      x: 0,
      previousX: 0,
      velocityX: 0,
      goalX: 0,
      y: 0,
      previousY: 0,
      velocityY: 0,
      goalY: 0,
    };

    let nuclei = [];
    let scanPlan = [];
    let queue = [];
    let state = STATES.PRELOAD;
    let stateElapsed = 0;
    let storyElapsed = 0;
    let introProgress = 0;
    let patchReveal = 0;
    let siblingReveal = 0;
    let fieldDim = 1;
    let logoRotation = 0;
    let previousLogoRotation = 0;
    let logoColorProgress = 0;
    let previousLogoColorProgress = 0;
    let logoDetailAlpha = 1;
    let previousLogoDetailAlpha = 1;
    let loopDiveScale = 1;
    let previousLoopDiveScale = 1;
    let loopBlackAlpha = 0;
    let previousLoopBlackAlpha = 0;
    let loopFieldReset = false;
    let activeNucleusId = -1;
    let currentScanIndex = -1;
    let scanSubphase = "IDLE";
    let scanSubphaseElapsed = 0;
    let scanSubphaseDuration = 1;
    let completed = false;
    let cycleCount = 0;

    const auditFrame = {
      visible: false,
      x: 0,
      previousX: 0,
      y: 0,
      previousY: 0,
      size: 40,
      previousSize: 40,
      fromX: 0,
      fromY: 0,
      fromSize: 40,
      controlOneX: 0,
      controlOneY: 0,
      controlTwoX: 0,
      controlTwoY: 0,
      toX: 0,
      toY: 0,
      toSize: 40,
      alpha: 0,
      previousAlpha: 0,
      settleAmount: 0,
    };

    const flyingCopy = {
      active: false,
      sourceId: -1,
      slot: -1,
      phase: "IDLE",
      elapsed: 0,
      duration: 1,
      x: 0,
      previousX: 0,
      y: 0,
      previousY: 0,
      heightScale: 1,
      previousHeightScale: 1,
      alpha: 0,
      previousAlpha: 0,
      fromX: 0,
      fromY: 0,
      controlOneX: 0,
      controlOneY: 0,
      controlTwoX: 0,
      controlTwoY: 0,
      toX: 0,
      toY: 0,
    };

    let animationFrame = 0;
    let resizeFrame = 0;
    let running = false;
    let intersecting = false;
    let pageVisible = !document.hidden;
    let previousFrameTimestamp = 0;
    let accumulator = 0;
    let renderAccumulator = 0;
    let renderedFrameCount = 0;

    function setState(nextState) {
      state = nextState;
      stateElapsed = 0;
      canvas.dataset.state = nextState;
    }

    function measure() {
      const canvasRect = canvas.getBoundingClientRect();
      width = Math.max(1, Math.round(canvasRect.width));
      height = Math.max(1, Math.round(canvasRect.height));
      isMobile = width <= CONFIG.breakpoint;
      layout = isMobile ? CONFIG.mobile : CONFIG.desktop;
      const deviceDpr = window.devicePixelRatio || 1;
      const budgetDpr = Math.sqrt(renderProfile.pixelBudget / Math.max(1, width * height));
      dpr = Math.max(0.75, Math.min(deviceDpr, renderProfile.dprCap, budgetDpr));
      const pixelWidth = Math.max(1, Math.round(width * dpr));
      const pixelHeight = Math.max(1, Math.round(height * dpr));
      if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
      if (canvas.height !== pixelHeight) canvas.height = pixelHeight;
      canvas.dataset.quality = qualityTier;
      canvas.dataset.renderDpr = dpr.toFixed(2);
      patchSize = Math.min(width, height) * layout.patchScale;
      // The AANCA mark uses 8 px tiles with a 3 px gap: 1 + 3 / 8.
      gridSpacing = patchSize * 1.375;
      const halfSpacing = gridSpacing * 0.5;

      patchCenters[0] = -halfSpacing;
      patchCenters[1] = -halfSpacing;
      patchCenters[2] = halfSpacing;
      patchCenters[3] = -halfSpacing;
      patchCenters[4] = -halfSpacing;
      patchCenters[5] = halfSpacing;
      patchCenters[6] = halfSpacing;
      patchCenters[7] = halfSpacing;

      const studyPatchPixels = clamp(
        width * layout.studyPatchFraction,
        layout.studyPatchMin,
        layout.studyPatchMax
      );
      const logoPatchPixels = clamp(
        width * layout.logoPatchFraction,
        layout.logoPatchMin,
        layout.logoPatchMax
      );

      const viewportCenterX = width * 0.5;
      const viewportCenterY = height * 0.5;
      const initialPatchScreenX = width * layout.patchScreenX;
      const initialPatchScreenY = height * layout.patchScreenY;

      initialCamera.scale = 1;
      initialCamera.x = initialPatchScreenX - viewportCenterX - patchCenters[0];
      initialCamera.y = initialPatchScreenY - viewportCenterY - patchCenters[1];

      patchCamera.scale = isMobile ? 0.8 : 0.76;
      patchCamera.x =
        initialPatchScreenX - viewportCenterX - patchCenters[0] * patchCamera.scale;
      patchCamera.y =
        initialPatchScreenY - viewportCenterY - patchCenters[1] * patchCamera.scale;

      studyCamera.scale = studyPatchPixels / patchSize;
      studyCamera.x = width * layout.studyScreenX - viewportCenterX;
      studyCamera.y = height * layout.studyScreenY - viewportCenterY;

      finalCamera.scale = logoPatchPixels / patchSize;
      finalCamera.x = width * layout.logoScreenX - viewportCenterX;
      finalCamera.y = height * layout.logoScreenY - viewportCenterY;

      if (heroCopy) {
        const copyRect = heroCopy.getBoundingClientRect();
        textSafeLeft = copyRect.left - canvasRect.left - layout.textClearance;
        textSafeTop = copyRect.top - canvasRect.top - layout.textClearance;
        textSafeRight = copyRect.right - canvasRect.left + layout.textClearance;
        textSafeBottom = copyRect.bottom - canvasRect.top + layout.textClearance;
      } else {
        textSafeLeft = -1;
        textSafeTop = -1;
        textSafeRight = -1;
        textSafeBottom = -1;
      }
    }

    function resetCamera(position) {
      camera.scale = position.scale;
      camera.previousScale = position.scale;
      camera.velocityScale = 0;
      camera.goalScale = position.scale;
      camera.x = position.x;
      camera.previousX = position.x;
      camera.velocityX = 0;
      camera.goalX = position.x;
      camera.y = position.y;
      camera.previousY = position.y;
      camera.velocityY = 0;
      camera.goalY = position.y;
    }

    function queuePosition(slot, output) {
      if (layout.queueHorizontal) {
        const spacing = 0.22;
        const start = -spacing * (layout.selectCount - 1) * 0.5;
        output.x = patchCenters[0] + (start + slot * spacing) * patchSize;
        output.y = patchCenters[1] + 0.4 * patchSize;
        output.scale = 0.72;
      } else {
        const spacing = 0.145;
        output.x = patchCenters[0] + 0.43 * patchSize;
        output.y =
          patchCenters[1] + (-spacing * (layout.selectCount - 1) * 0.5 + slot * spacing) * patchSize;
        output.scale = 0.76;
      }
    }

    const reusableQueuePosition = { x: 0, y: 0, scale: 1 };

    function candidateOverlapsText(localX, localY, radius) {
      if (textSafeLeft < 0) return false;
      const screenX = width * layout.patchScreenX + localX * patchSize;
      const screenY = height * layout.patchScreenY + localY * patchSize;
      const pixelRadius = radius * patchSize;
      return !(
        screenX + pixelRadius < textSafeLeft ||
        screenX - pixelRadius > textSafeRight ||
        screenY + pixelRadius < textSafeTop ||
        screenY - pixelRadius > textSafeBottom
      );
    }

    function generateNuclei() {
      const random = createSeededRandom(CONFIG.seed ^ (isMobile ? 0x93a1 : 0x2fc7));
      const generated = [];
      let attempts = 0;
      const maximumAttempts = layout.nucleusCount * 900;

      while (generated.length < layout.nucleusCount && attempts < maximumAttempts) {
        const index = generated.length;
        const regionIndex = (index + attempts) % REGIONS.length;
        const region = REGIONS[regionIndex];
        attempts += 1;

        const angle = random() * Math.PI * 2;
        const radial = Math.sqrt(random());
        const localX = region.x + Math.cos(angle) * radial * region.rx;
        const localY = region.y + Math.sin(angle) * radial * region.ry;
        const assetIndex = Math.floor(random() * sprites.length);
        const scale = randomRange(random, 0.55, 1.15);
        const stretchX = randomRange(random, 0.94, 1.06);
        const stretchY = randomRange(random, 0.96, 1.04);
        const heightNorm = layout.baseNucleusHeight * scale;
        const widthNorm = heightNorm * sprites[assetIndex].aspect * stretchX;
        const radius = Math.max(widthNorm, heightNorm * stretchY) * 0.34;

        if (candidateOverlapsText(localX, localY, radius)) continue;

        let separated = true;
        for (let otherIndex = 0; otherIndex < generated.length; otherIndex += 1) {
          const other = generated[otherIndex];
          const distance = Math.hypot(localX - other.localX, localY - other.localY);
          if (distance < (radius + other.radius) * 0.72) {
            separated = false;
            break;
          }
        }
        if (!separated) continue;

        generated.push({
          id: index,
          assetIndex,
          region: regionIndex,
          localX,
          localY,
          heightNorm,
          radius,
          rotation: randomRange(random, -0.24, 0.24),
          flip: random() < 0.42 ? -1 : 1,
          stretchX,
          stretchY,
          baseAlpha: randomRange(random, 0.48, 0.62),
          introDelay:
            (Math.floor(random() * CONFIG.fieldReveal.groupCount) /
              (CONFIG.fieldReveal.groupCount - 1)) *
            CONFIG.fieldReveal.staggerSpan,
          selectedPlanned: false,
          selected: false,
          queued: false,
          markProgress: 0,
        });
      }

      return generated;
    }

    function buildScanPlan() {
      const random = createSeededRandom(CONFIG.seed ^ (isMobile ? 0x51e7 : 0xc021));
      const regionBuckets = [[], [], [], []];
      for (let index = 0; index < nuclei.length; index += 1) {
        regionBuckets[nuclei[index].region].push(index);
      }
      for (let index = 0; index < regionBuckets.length; index += 1) {
        shuffleInPlace(regionBuckets[index], random);
      }

      const plan = [];
      const selectedPositions = new Set(layout.selectedScanPositions);
      for (let index = 0; index < layout.auditCount; index += 1) {
        let bucket = regionBuckets[index % regionBuckets.length];
        if (bucket.length === 0) {
          for (let bucketIndex = 0; bucketIndex < regionBuckets.length; bucketIndex += 1) {
            if (regionBuckets[bucketIndex].length > bucket.length) {
              bucket = regionBuckets[bucketIndex];
            }
          }
        }
        const nucleusId = bucket.pop();
        if (typeof nucleusId !== "number") continue;
        const selected = selectedPositions.has(index);
        nuclei[nucleusId].selectedPlanned = selected;
        plan.push({
          nucleusId,
          selected,
          travelDuration: randomRange(
            random,
            CONFIG.timings.frameTravelMin,
            CONFIG.timings.frameTravelMax
          ),
          settleDuration: randomRange(
            random,
            CONFIG.timings.frameSettleMin,
            CONFIG.timings.frameSettleMax
          ),
          inspectDuration: randomRange(
            random,
            CONFIG.timings.inspectMin,
            CONFIG.timings.inspectMax
          ),
          flightDuration: randomRange(
            random,
            CONFIG.timings.copyFlightMin,
            CONFIG.timings.copyFlightMax
          ),
          bend: randomRange(random, -1, 1),
        });
      }
      return plan;
    }

    function drawSprite(targetContext, nucleus, x, y, visualHeight, alpha) {
      if (!(visualHeight > 0 && alpha > 0.001)) return;
      const sprite = sprites[nucleus.assetIndex];
      const drawHeight = visualHeight * nucleus.stretchY;
      const drawWidth = visualHeight * sprite.aspect * nucleus.stretchX;

      targetContext.save();
      targetContext.translate(x, y);
      targetContext.rotate(nucleus.rotation);
      targetContext.scale(nucleus.flip, 1);
      targetContext.globalAlpha = clamp(alpha, 0, 1);
      if (sprite.source) {
        targetContext.drawImage(
          sprite.source,
          -drawWidth * 0.5,
          -drawHeight * 0.5,
          drawWidth,
          drawHeight
        );
      } else {
        targetContext.scale(drawWidth, drawHeight);
        targetContext.fillStyle = COLORS.body;
        targetContext.fill(FALLBACK_PATH);
        targetContext.strokeStyle = COLORS.annotation;
        targetContext.lineWidth = 0.018;
        targetContext.stroke(FALLBACK_PATH);
      }
      targetContext.restore();
    }

    function drawCorners(targetContext, x, y, size, alpha, cameraScale, progress) {
      if (alpha <= 0.001 || progress <= 0.001) return;
      const half = size * 0.5;
      const inset = Math.min(size * 0.12, 7 / Math.max(0.2, cameraScale));
      const length = (10 + clamp(size * cameraScale * 0.05, 0, 4)) / Math.max(0.2, cameraScale);
      const drawnLength = length * easeOutCubic(clamp(progress, 0, 1));

      targetContext.save();
      targetContext.globalAlpha = clamp(alpha, 0, 1);
      targetContext.strokeStyle = COLORS.accentBright;
      targetContext.lineWidth = Math.min(2.5, 1.25 / Math.max(0.2, cameraScale));
      targetContext.lineCap = "round";
      targetContext.beginPath();
      targetContext.moveTo(x - half + inset, y - half + inset + drawnLength);
      targetContext.lineTo(x - half + inset, y - half + inset);
      targetContext.lineTo(x - half + inset + drawnLength, y - half + inset);
      targetContext.moveTo(x + half - inset - drawnLength, y - half + inset);
      targetContext.lineTo(x + half - inset, y - half + inset);
      targetContext.lineTo(x + half - inset, y - half + inset + drawnLength);
      targetContext.moveTo(x + half - inset, y + half - inset - drawnLength);
      targetContext.lineTo(x + half - inset, y + half - inset);
      targetContext.lineTo(x + half - inset - drawnLength, y + half - inset);
      targetContext.moveTo(x - half + inset + drawnLength, y + half - inset);
      targetContext.lineTo(x - half + inset, y + half - inset);
      targetContext.lineTo(x - half + inset, y + half - inset - drawnLength);
      targetContext.stroke();
      targetContext.restore();
    }

    function buildPatchCaches() {
      patchCaches = [];
      const cacheSize = renderProfile.cacheSize;
      for (let patchIndex = 1; patchIndex < 4; patchIndex += 1) {
        const cache = document.createElement("canvas");
        cache.width = cacheSize;
        cache.height = cacheSize;
        const cacheContext = cache.getContext("2d", { alpha: true });
        if (!cacheContext) continue;
        const random = createSeededRandom(CONFIG.seed ^ (patchIndex * 0x1f31));
        const cachedNuclei = [];
        let attempts = 0;
        while (cachedNuclei.length < 34 && attempts < 6000) {
          attempts += 1;
          const region = REGIONS[(cachedNuclei.length + attempts) % REGIONS.length];
          const angle = random() * Math.PI * 2;
          const radial = Math.sqrt(random());
          const localX = region.x + Math.cos(angle) * radial * region.rx;
          const localY = region.y + Math.sin(angle) * radial * region.ry;
          const assetIndex = Math.floor(random() * sprites.length);
          const heightNorm = randomRange(random, 0.052, 0.094);
          const radius = Math.max(heightNorm, heightNorm * sprites[assetIndex].aspect) * 0.33;
          let separated = true;
          for (let otherIndex = 0; otherIndex < cachedNuclei.length; otherIndex += 1) {
            const other = cachedNuclei[otherIndex];
            if (
              Math.hypot(localX - other.localX, localY - other.localY) <
              (radius + other.radius) * 0.62
            ) {
              separated = false;
              break;
            }
          }
          if (!separated) continue;
          cachedNuclei.push({
            assetIndex,
            localX,
            localY,
            heightNorm,
            radius,
            rotation: randomRange(random, -0.24, 0.24),
            flip: random() < 0.42 ? -1 : 1,
            stretchX: randomRange(random, 0.94, 1.06),
            stretchY: randomRange(random, 0.96, 1.04),
          });
        }

        for (let index = 0; index < cachedNuclei.length; index += 1) {
          const nucleus = cachedNuclei[index];
          const x = cacheSize * (0.5 + nucleus.localX);
          const y = cacheSize * (0.5 + nucleus.localY);
          const selected = index === 7 || index === 23;
          drawSprite(
            cacheContext,
            nucleus,
            x,
            y,
            nucleus.heightNorm * cacheSize,
            selected ? 0.9 : 0.62
          );
          if (selected) {
            const sprite = sprites[nucleus.assetIndex];
            const markerSize =
              Math.max(
                nucleus.heightNorm,
                nucleus.heightNorm * sprite.aspect * nucleus.stretchX
              ) *
                cacheSize +
              10;
            drawCorners(cacheContext, x, y, markerSize, 0.88, 1, 1);
          }
        }
        patchCaches.push(cache);
      }
    }

    function clearReviewState() {
      queue = [];
      currentScanIndex = -1;
      scanSubphase = "IDLE";
      scanSubphaseElapsed = 0;
      activeNucleusId = -1;
      auditFrame.visible = false;
      auditFrame.alpha = 0;
      auditFrame.previousAlpha = 0;
      flyingCopy.active = false;
      flyingCopy.phase = "IDLE";
      for (let index = 0; index < nuclei.length; index += 1) {
        nuclei[index].selected = false;
        nuclei[index].queued = false;
        nuclei[index].markProgress = 0;
      }
    }

    function resetStoryData() {
      clearReviewState();
      introProgress = 0;
      patchReveal = 0;
      siblingReveal = 0;
      fieldDim = 1;
      logoRotation = 0;
      previousLogoRotation = 0;
      logoColorProgress = 0;
      previousLogoColorProgress = 0;
      logoDetailAlpha = 1;
      previousLogoDetailAlpha = 1;
      loopDiveScale = 1;
      previousLoopDiveScale = 1;
      loopBlackAlpha = 0;
      previousLoopBlackAlpha = 0;
      loopFieldReset = false;
      storyElapsed = 0;
      completed = false;
      cycleCount = 0;
      canvas.dataset.complete = "false";
      canvas.dataset.cycle = "0";
    }

    function rebuildScene() {
      nuclei = generateNuclei();
      scanPlan = buildScanPlan();
      if (patchCaches.length !== 3) buildPatchCaches();
      resetStoryData();
      resetCamera(initialCamera);
    }

    function startIntro() {
      resetStoryData();
      resetCamera(initialCamera);
      setState(STATES.INTRO);
      renderScene(1);
    }

    function currentPatchWorldX() {
      return patchCenters[0];
    }

    function currentPatchWorldY() {
      return patchCenters[1];
    }

    function nucleusWorldX(nucleus) {
      return currentPatchWorldX() + nucleus.localX * patchSize;
    }

    function nucleusWorldY(nucleus) {
      return currentPatchWorldY() + nucleus.localY * patchSize;
    }

    function nucleusVisualHeight(nucleus) {
      return nucleus.heightNorm * patchSize;
    }

    function nucleusFrameSize(nucleus) {
      const sprite = sprites[nucleus.assetIndex];
      const visualHeight = nucleusVisualHeight(nucleus) * nucleus.stretchY;
      const visualWidth = nucleusVisualHeight(nucleus) * sprite.aspect * nucleus.stretchX;
      return Math.max(visualWidth, visualHeight) + 16;
    }

    function beginScan(scanIndex) {
      if (scanIndex >= scanPlan.length || queue.length >= layout.selectCount) {
        activeNucleusId = -1;
        setState(STATES.QUEUE_COMPLETE);
        return;
      }

      currentScanIndex = scanIndex;
      const plan = scanPlan[scanIndex];
      const nucleus = nuclei[plan.nucleusId];
      const toX = nucleusWorldX(nucleus);
      const toY = nucleusWorldY(nucleus);
      const toSize = nucleusFrameSize(nucleus);
      const fromX = auditFrame.visible
        ? auditFrame.x
        : currentPatchWorldX() - patchSize * 0.38;
      const fromY = auditFrame.visible
        ? auditFrame.y
        : currentPatchWorldY() - patchSize * 0.04;
      const fromSize = auditFrame.visible ? auditFrame.size : toSize * 0.88;
      const deltaX = toX - fromX;
      const deltaY = toY - fromY;
      const distance = Math.hypot(deltaX, deltaY) || 1;
      const perpendicularX = (-deltaY / distance) * patchSize * 0.07 * plan.bend;
      const perpendicularY = (deltaX / distance) * patchSize * 0.07 * plan.bend;

      auditFrame.visible = true;
      auditFrame.fromX = fromX;
      auditFrame.fromY = fromY;
      auditFrame.fromSize = fromSize;
      auditFrame.toX = toX;
      auditFrame.toY = toY;
      auditFrame.toSize = toSize;
      auditFrame.controlOneX = fromX + deltaX * 0.3 + perpendicularX;
      auditFrame.controlOneY = fromY + deltaY * 0.3 + perpendicularY;
      auditFrame.controlTwoX = fromX + deltaX * 0.72 - perpendicularX * 0.45;
      auditFrame.controlTwoY = fromY + deltaY * 0.72 - perpendicularY * 0.45;
      auditFrame.x = fromX;
      auditFrame.previousX = fromX;
      auditFrame.y = fromY;
      auditFrame.previousY = fromY;
      auditFrame.size = fromSize;
      auditFrame.previousSize = fromSize;
      auditFrame.alpha = scanIndex === 0 ? 0 : 1;
      auditFrame.previousAlpha = auditFrame.alpha;
      auditFrame.settleAmount = 0;
      activeNucleusId = plan.nucleusId;
      scanSubphase = "TRAVEL";
      scanSubphaseElapsed = 0;
      scanSubphaseDuration = plan.travelDuration;
      setState(STATES.SCAN);
    }

    function beginClone() {
      const plan = scanPlan[currentScanIndex];
      const nucleus = nuclei[plan.nucleusId];
      nucleus.selected = true;
      nucleus.markProgress = 0;
      flyingCopy.active = false;
      flyingCopy.sourceId = nucleus.id;
      flyingCopy.slot = queue.length;
      flyingCopy.phase = "MARK";
      flyingCopy.elapsed = 0;
      flyingCopy.duration = CONFIG.timings.mark;
      setState(STATES.SELECT_AND_CLONE);
    }

    function beginCopyFlight() {
      const plan = scanPlan[currentScanIndex];
      const nucleus = nuclei[flyingCopy.sourceId];
      queuePosition(flyingCopy.slot, reusableQueuePosition);
      const fromX = nucleusWorldX(nucleus);
      const fromY = nucleusWorldY(nucleus);
      const toX = reusableQueuePosition.x;
      const toY = reusableQueuePosition.y;
      const deltaX = toX - fromX;
      const deltaY = toY - fromY;
      const distance = Math.hypot(deltaX, deltaY) || 1;
      const bend = patchSize * 0.075 * (flyingCopy.slot % 2 === 0 ? 1 : -1);
      const perpendicularX = (-deltaY / distance) * bend;
      const perpendicularY = (deltaX / distance) * bend;

      flyingCopy.active = true;
      flyingCopy.phase = "FLIGHT";
      flyingCopy.elapsed = 0;
      flyingCopy.duration = plan.flightDuration;
      flyingCopy.fromX = fromX;
      flyingCopy.fromY = fromY;
      flyingCopy.toX = toX;
      flyingCopy.toY = toY;
      flyingCopy.controlOneX = fromX + deltaX * 0.3 + perpendicularX;
      flyingCopy.controlOneY = fromY + deltaY * 0.3 + perpendicularY;
      flyingCopy.controlTwoX = fromX + deltaX * 0.74 - perpendicularX * 0.38;
      flyingCopy.controlTwoY = fromY + deltaY * 0.74 - perpendicularY * 0.38;
      flyingCopy.x = fromX;
      flyingCopy.previousX = fromX;
      flyingCopy.y = fromY;
      flyingCopy.previousY = fromY;
      flyingCopy.heightScale = 1;
      flyingCopy.previousHeightScale = 1;
      flyingCopy.alpha = 0.9;
      flyingCopy.previousAlpha = 0.9;
    }

    function finishCopyFlight() {
      const nucleus = nuclei[flyingCopy.sourceId];
      nucleus.queued = true;
      queue.push({ nucleusId: nucleus.id, slot: flyingCopy.slot });
      flyingCopy.active = false;
      flyingCopy.phase = "IDLE";
      beginScan(currentScanIndex + 1);
    }

    function beginZoomToPatch() {
      zoomStartCamera.scale = camera.goalScale;
      zoomStartCamera.x = camera.goalX;
      zoomStartCamera.y = camera.goalY;
      activeNucleusId = -1;
      setState(STATES.ZOOM_TO_PATCH);
    }

    function beginZoomToStudy() {
      zoomStartCamera.scale = camera.goalScale;
      zoomStartCamera.x = camera.goalX;
      zoomStartCamera.y = camera.goalY;
      setState(STATES.ZOOM_TO_STUDY);
    }

    function beginReturnToField() {
      clearReviewState();
      introProgress = 1;
      patchReveal = 1;
      siblingReveal = 1;
      fieldDim = 0.78;
      logoRotation = Math.PI / 4;
      logoColorProgress = 1;
      logoDetailAlpha = 0;
      loopDiveScale = 1;
      loopBlackAlpha = 0;
      loopFieldReset = false;
      camera.goalScale = finalCamera.scale;
      camera.goalX = finalCamera.x;
      camera.goalY = finalCamera.y;
      cycleCount += 1;
      canvas.dataset.cycle = String(cycleCount);
      setState(STATES.RETURN_TO_FIELD);
    }

    function restartLoopCycle() {
      storyElapsed = 0;
      introProgress = 1;
      patchReveal = 0;
      siblingReveal = 0;
      fieldDim = 1;
      logoRotation = 0;
      logoColorProgress = 0;
      logoDetailAlpha = 1;
      loopDiveScale = 1;
      loopBlackAlpha = 0;
      loopFieldReset = false;
      camera.goalScale = initialCamera.scale;
      camera.goalX = initialCamera.x;
      camera.goalY = initialCamera.y;
      beginScan(0);
    }

    function snapshotInterpolants() {
      camera.previousScale = camera.scale;
      camera.previousX = camera.x;
      camera.previousY = camera.y;
      auditFrame.previousX = auditFrame.x;
      auditFrame.previousY = auditFrame.y;
      auditFrame.previousSize = auditFrame.size;
      auditFrame.previousAlpha = auditFrame.alpha;
      flyingCopy.previousX = flyingCopy.x;
      flyingCopy.previousY = flyingCopy.y;
      flyingCopy.previousHeightScale = flyingCopy.heightScale;
      flyingCopy.previousAlpha = flyingCopy.alpha;
      previousLogoRotation = logoRotation;
      previousLogoColorProgress = logoColorProgress;
      previousLogoDetailAlpha = logoDetailAlpha;
      previousLoopDiveScale = loopDiveScale;
      previousLoopBlackAlpha = loopBlackAlpha;
    }

    function updateScan(deltaSeconds) {
      const plan = scanPlan[currentScanIndex];
      if (!plan) {
        setState(STATES.QUEUE_COMPLETE);
        return;
      }

      scanSubphaseElapsed += deltaSeconds;
      const rawProgress = clamp(scanSubphaseElapsed / scanSubphaseDuration, 0, 1);

      if (scanSubphase === "TRAVEL") {
        const progress = easeInOutCubic(rawProgress);
        auditFrame.x = cubicBezier(
          auditFrame.fromX,
          auditFrame.controlOneX,
          auditFrame.controlTwoX,
          auditFrame.toX,
          progress
        );
        auditFrame.y = cubicBezier(
          auditFrame.fromY,
          auditFrame.controlOneY,
          auditFrame.controlTwoY,
          auditFrame.toY,
          progress
        );
        auditFrame.size = lerp(auditFrame.fromSize, auditFrame.toSize, progress);
        auditFrame.alpha = Math.min(1, rawProgress * 3.4);
        if (rawProgress >= 1) {
          auditFrame.x = auditFrame.toX;
          auditFrame.y = auditFrame.toY;
          auditFrame.size = auditFrame.toSize;
          scanSubphase = "SETTLE";
          scanSubphaseElapsed = 0;
          scanSubphaseDuration = plan.settleDuration;
        }
        return;
      }

      if (scanSubphase === "SETTLE") {
        auditFrame.settleAmount = easeOutCubic(rawProgress);
        auditFrame.size = auditFrame.toSize * lerp(1.035, 1, auditFrame.settleAmount);
        auditFrame.alpha = 1;
        if (rawProgress >= 1) {
          auditFrame.size = auditFrame.toSize;
          scanSubphase = "INSPECT";
          scanSubphaseElapsed = 0;
          scanSubphaseDuration = plan.inspectDuration;
        }
        return;
      }

      auditFrame.alpha = 1;
      if (rawProgress >= 1) {
        if (plan.selected && !nuclei[plan.nucleusId].selected) beginClone();
        else beginScan(currentScanIndex + 1);
      }
    }

    function updateClone(deltaSeconds) {
      const nucleus = nuclei[flyingCopy.sourceId];
      if (!nucleus) {
        beginScan(currentScanIndex + 1);
        return;
      }

      flyingCopy.elapsed += deltaSeconds;
      const rawProgress = clamp(flyingCopy.elapsed / flyingCopy.duration, 0, 1);
      if (flyingCopy.phase === "MARK") {
        nucleus.markProgress = easeOutCubic(rawProgress);
        auditFrame.alpha = lerp(1, 0.68, rawProgress);
        if (rawProgress >= 1) beginCopyFlight();
        return;
      }

      if (flyingCopy.phase === "FLIGHT") {
        const progress = easeInOutQuint(rawProgress);
        flyingCopy.x = cubicBezier(
          flyingCopy.fromX,
          flyingCopy.controlOneX,
          flyingCopy.controlTwoX,
          flyingCopy.toX,
          progress
        );
        flyingCopy.y = cubicBezier(
          flyingCopy.fromY,
          flyingCopy.controlOneY,
          flyingCopy.controlTwoY,
          flyingCopy.toY,
          progress
        );
        queuePosition(flyingCopy.slot, reusableQueuePosition);
        flyingCopy.heightScale = lerp(1, reusableQueuePosition.scale, progress);
        flyingCopy.alpha = lerp(0.9, 1, easeOutCubic(rawProgress));
        auditFrame.alpha = lerp(0.68, 0.32, rawProgress);
        if (rawProgress >= 1) finishCopyFlight();
      }
    }

    function updateSimulation(deltaSeconds) {
      stateElapsed += deltaSeconds;
      storyElapsed += deltaSeconds;

      if (state === STATES.INTRO) {
        introProgress = easeOutCubic(clamp(stateElapsed / CONFIG.timings.intro, 0, 1));
        if (stateElapsed >= CONFIG.timings.intro) {
          introProgress = 1;
          beginScan(0);
        }
        return;
      }

      if (state === STATES.SCAN) {
        updateScan(deltaSeconds);
        return;
      }

      if (state === STATES.SELECT_AND_CLONE) {
        updateClone(deltaSeconds);
        return;
      }

      if (state === STATES.QUEUE_COMPLETE) {
        const progress = clamp(stateElapsed / CONFIG.timings.queueComplete, 0, 1);
        auditFrame.alpha = lerp(0.32, 0, easeOutCubic(progress));
        fieldDim = lerp(1, 0.9, easeOutCubic(progress));
        if (stateElapsed >= CONFIG.timings.queueComplete) beginZoomToPatch();
        return;
      }

      if (state === STATES.ZOOM_TO_PATCH) {
        const rawProgress = clamp(stateElapsed / CONFIG.timings.zoomToPatch, 0, 1);
        const progress = easeInOutCubic(rawProgress);
        patchReveal = progress;
        fieldDim = lerp(0.9, 0.82, progress);
        camera.goalScale = lerp(zoomStartCamera.scale, patchCamera.scale, progress);
        camera.goalX = lerp(zoomStartCamera.x, patchCamera.x, progress);
        camera.goalY = lerp(zoomStartCamera.y, patchCamera.y, progress);
        if (rawProgress >= 1) beginZoomToStudy();
        return;
      }

      if (state === STATES.ZOOM_TO_STUDY) {
        const rawProgress = clamp(stateElapsed / CONFIG.timings.zoomToStudy, 0, 1);
        const cameraProgress = easeInOutCubic(
          clamp(rawProgress / CONFIG.morph.cameraEnd, 0, 1)
        );
        const siblingProgress = easeInOutCubic(
          clamp(
            (rawProgress - CONFIG.morph.siblingStart) /
              (CONFIG.morph.siblingEnd - CONFIG.morph.siblingStart),
            0,
            1
          )
        );
        const rotateProgress = easeInOutCubic(
          clamp(
            (rawProgress - CONFIG.morph.rotateStart) /
              (CONFIG.morph.rotateEnd - CONFIG.morph.rotateStart),
            0,
            1
          )
        );
        const colorProgress = easeInOutCubic(
          clamp(
            (rawProgress - CONFIG.morph.colorStart) /
              (CONFIG.morph.colorEnd - CONFIG.morph.colorStart),
            0,
            1
          )
        );
        patchReveal = 1;
        siblingReveal = siblingProgress;
        fieldDim = lerp(0.82, 0.78, cameraProgress);
        camera.goalScale = cubicBezier(
          zoomStartCamera.scale,
          studyCamera.scale,
          studyCamera.scale,
          finalCamera.scale,
          cameraProgress
        );
        camera.goalX = cubicBezier(
          zoomStartCamera.x,
          studyCamera.x,
          studyCamera.x,
          finalCamera.x,
          cameraProgress
        );
        camera.goalY = cubicBezier(
          zoomStartCamera.y,
          studyCamera.y,
          studyCamera.y,
          finalCamera.y,
          cameraProgress
        );
        logoRotation = (Math.PI / 4) * rotateProgress;
        logoColorProgress = colorProgress;
        logoDetailAlpha = 1 - colorProgress;
        if (rawProgress >= 1) {
          camera.goalScale = finalCamera.scale;
          camera.goalX = finalCamera.x;
          camera.goalY = finalCamera.y;
          siblingReveal = 1;
          logoRotation = Math.PI / 4;
          logoColorProgress = 1;
          logoDetailAlpha = 0;
          setState(STATES.SETTLED);
        }
        return;
      }

      if (state === STATES.SETTLED) {
        camera.goalScale = finalCamera.scale;
        camera.goalX = finalCamera.x;
        camera.goalY = finalCamera.y;
        patchReveal = 1;
        siblingReveal = 1;
        logoRotation = Math.PI / 4;
        logoColorProgress = 1;
        logoDetailAlpha = 0;

        if (stateElapsed >= CONFIG.timings.logoDisplay) beginReturnToField();
        return;
      }

      if (state === STATES.RETURN_TO_FIELD) {
        const spinDive = CONFIG.timings.returnSpinDive;
        const fadeToBlack = CONFIG.timings.returnFadeToBlack;
        const blackHold = CONFIG.timings.returnBlackHold;
        const revealField = CONFIG.timings.returnRevealField;
        const fadeComplete = spinDive + fadeToBlack;
        const revealStart = fadeComplete + blackHold;
        const revealComplete = revealStart + revealField;
        const sequenceDuration = revealComplete + CONFIG.timings.returnFieldHold;

        if (stateElapsed < spinDive) {
          const rawProgress = clamp(stateElapsed / spinDive, 0, 1);
          const diveProgress = easeInOutCubic(rawProgress);
          camera.goalScale = finalCamera.scale;
          camera.goalX = finalCamera.x;
          camera.goalY = finalCamera.y;
          logoRotation = Math.PI / 4 + diveProgress * Math.PI * 4.5;
          logoColorProgress = 1;
          logoDetailAlpha = 0;
          loopDiveScale = lerp(1, CONFIG.dive.endScale, diveProgress);
          loopBlackAlpha = 0;
          patchReveal = 1;
          siblingReveal = 1;
          fieldDim = 0.78;
          return;
        }

        if (stateElapsed < fadeComplete) {
          const fadeProgress = easeInOutCubic(
            clamp((stateElapsed - spinDive) / fadeToBlack, 0, 1)
          );
          logoRotation = Math.PI / 4 + Math.PI * (4.5 + fadeProgress * 0.75);
          logoColorProgress = 1;
          logoDetailAlpha = 0;
          loopDiveScale = lerp(
            CONFIG.dive.endScale,
            CONFIG.dive.fadeScale,
            fadeProgress
          );
          loopBlackAlpha = fadeProgress;
          return;
        }

        if (!loopFieldReset) {
          resetCamera(initialCamera);
          introProgress = 0;
          loopFieldReset = true;
        }
        camera.goalScale = initialCamera.scale;
        camera.goalX = initialCamera.x;
        camera.goalY = initialCamera.y;
        logoRotation = 0;
        logoColorProgress = 0;
        logoDetailAlpha = 1;
        loopDiveScale = 1;
        siblingReveal = 0;
        patchReveal = 0;
        fieldDim = 1;

        if (stateElapsed < revealStart) {
          loopBlackAlpha = 1;
          return;
        }

        const revealRawProgress = clamp(
          (stateElapsed - revealStart) / revealField,
          0,
          1
        );
        introProgress = revealRawProgress;
        const revealProgress = easeInOutCubic(
          clamp(revealRawProgress / CONFIG.fieldReveal.blackReleaseEnd, 0, 1)
        );
        loopBlackAlpha = 1 - revealProgress;

        if (stateElapsed >= sequenceDuration) restartLoopCycle();
        return;
      }
    }

    function updateCamera(deltaSeconds) {
      const steps = Math.max(1, Math.ceil(deltaSeconds / CONFIG.cameraSubstepSeconds));
      const step = deltaSeconds / steps;
      for (let index = 0; index < steps; index += 1) {
        const scaleAcceleration =
          (camera.goalScale - camera.scale) * CONFIG.camera.stiffness -
          camera.velocityScale * CONFIG.camera.damping;
        camera.velocityScale += scaleAcceleration * step;
        camera.scale += camera.velocityScale * step;

        const xAcceleration =
          (camera.goalX - camera.x) * CONFIG.camera.stiffness -
          camera.velocityX * CONFIG.camera.damping;
        camera.velocityX += xAcceleration * step;
        camera.x += camera.velocityX * step;

        const yAcceleration =
          (camera.goalY - camera.y) * CONFIG.camera.stiffness -
          camera.velocityY * CONFIG.camera.damping;
        camera.velocityY += yAcceleration * step;
        camera.y += camera.velocityY * step;
      }
    }

    function cameraIsSettled() {
      return (
        Math.abs(camera.goalScale - camera.scale) < CONFIG.camera.scaleEpsilon &&
        Math.abs(camera.velocityScale) < CONFIG.camera.velocityEpsilon &&
        Math.abs(camera.goalX - camera.x) < CONFIG.camera.positionEpsilon &&
        Math.abs(camera.goalY - camera.y) < CONFIG.camera.positionEpsilon &&
        Math.abs(camera.velocityX) < CONFIG.camera.velocityEpsilon &&
        Math.abs(camera.velocityY) < CONFIG.camera.velocityEpsilon
      );
    }

    function drawPatchSurface(centerX, centerY, alpha, colorProgress) {
      if (alpha <= 0.001) return;
      const color = clamp(colorProgress, 0, 1);
      context.save();
      context.translate(centerX, centerY);
      context.scale(patchSize, patchSize);
      context.globalAlpha = alpha * 0.44 * (1 - color);
      context.fillStyle = COLORS.surface;
      context.fill(PATCH_PATH);
      context.globalAlpha = alpha * 0.62 * (1 - color);
      context.strokeStyle = COLORS.hairlineStrong;
      context.lineWidth = 1 / patchSize;
      context.stroke(PATCH_PATH);
      if (color > 0.001) {
        context.globalAlpha = alpha * color;
        context.fillStyle = COLORS.accent;
        context.fill(LOGO_TILE_PATH);
      }
      context.restore();
    }

    function clipCurrentPatch() {
      context.translate(currentPatchWorldX(), currentPatchWorldY());
      context.scale(patchSize, patchSize);
      context.clip(PATCH_PATH);
      context.scale(1 / patchSize, 1 / patchSize);
      context.translate(-currentPatchWorldX(), -currentPatchWorldY());
    }

    function introAmountFor(nucleus) {
      if (introProgress >= 1) return 1;
      return easeOutCubic(
        clamp((introProgress - nucleus.introDelay) / Math.max(0.1, 1 - nucleus.introDelay), 0, 1)
      );
    }

    function drawLiveField(renderedCameraScale, detailAlpha) {
      for (let index = 0; index < nuclei.length; index += 1) {
        const nucleus = nuclei[index];
        const introduction = introAmountFor(nucleus);
        if (introduction <= 0.001) continue;
        const x = nucleusWorldX(nucleus);
        const y = nucleusWorldY(nucleus);
        const selected = nucleus.selected;
        const active = nucleus.id === activeNucleusId;
        let alpha = nucleus.baseAlpha * fieldDim;
        if (active) alpha = 0.86;
        if (selected) alpha = lerp(0.82, 0.98, nucleus.markProgress);
        alpha *= introduction * layout.mobileAlpha * detailAlpha;
        const visualHeight = nucleusVisualHeight(nucleus) * lerp(0.965, 1, introduction);
        drawSprite(context, nucleus, x, y, visualHeight, alpha);
        if (selected) {
          drawCorners(
            context,
            x,
            y,
            nucleusFrameSize(nucleus),
            0.9 * layout.mobileAlpha * detailAlpha,
            renderedCameraScale,
            nucleus.markProgress
          );
        }
      }
    }

    function drawQueue(renderedCameraScale, detailAlpha) {
      if (queue.length === 0 || detailAlpha <= 0.001) return;

      queuePosition(0, reusableQueuePosition);
      const startX = reusableQueuePosition.x;
      const startY = reusableQueuePosition.y;
      queuePosition(queue.length - 1, reusableQueuePosition);
      context.save();
      context.globalAlpha = 0.24 * layout.mobileAlpha * detailAlpha;
      context.strokeStyle = COLORS.hairlineStrong;
      context.lineWidth = Math.min(2.2, 1 / Math.max(0.2, renderedCameraScale));
      context.beginPath();
      context.moveTo(startX, startY);
      context.lineTo(reusableQueuePosition.x, reusableQueuePosition.y);
      context.stroke();
      context.restore();

      for (let index = 0; index < queue.length; index += 1) {
        const item = queue[index];
        const nucleus = nuclei[item.nucleusId];
        queuePosition(item.slot, reusableQueuePosition);
        const visualHeight = nucleusVisualHeight(nucleus) * reusableQueuePosition.scale;
        drawSprite(
          context,
          nucleus,
          reusableQueuePosition.x,
          reusableQueuePosition.y,
          visualHeight,
          1 * layout.mobileAlpha * detailAlpha
        );
        drawCorners(
          context,
          reusableQueuePosition.x,
          reusableQueuePosition.y,
          nucleusFrameSize(nucleus) * reusableQueuePosition.scale,
          0.84 * layout.mobileAlpha * detailAlpha,
          renderedCameraScale,
          1
        );
      }
    }

    function drawFlyingCopy(interpolation, renderedCameraScale, detailAlpha) {
      if (!flyingCopy.active || detailAlpha <= 0.001) return;
      const nucleus = nuclei[flyingCopy.sourceId];
      const x = lerp(flyingCopy.previousX, flyingCopy.x, interpolation);
      const y = lerp(flyingCopy.previousY, flyingCopy.y, interpolation);
      const heightScale = lerp(
        flyingCopy.previousHeightScale,
        flyingCopy.heightScale,
        interpolation
      );
      const alpha = lerp(flyingCopy.previousAlpha, flyingCopy.alpha, interpolation);
      drawSprite(
        context,
        nucleus,
        x,
        y,
        nucleusVisualHeight(nucleus) * heightScale,
        alpha * detailAlpha
      );
      drawCorners(
        context,
        x,
        y,
        nucleusFrameSize(nucleus) * heightScale,
        0.84 * detailAlpha,
        renderedCameraScale,
        1
      );
    }

    function drawAuditFrame(interpolation, renderedCameraScale, detailAlpha) {
      if (!auditFrame.visible || auditFrame.alpha <= 0.001 || detailAlpha <= 0.001) return;
      const x = lerp(auditFrame.previousX, auditFrame.x, interpolation);
      const y = lerp(auditFrame.previousY, auditFrame.y, interpolation);
      const size = lerp(auditFrame.previousSize, auditFrame.size, interpolation);
      const alpha = lerp(auditFrame.previousAlpha, auditFrame.alpha, interpolation);
      drawCorners(context, x, y, size, alpha * detailAlpha, renderedCameraScale, 1);
    }

    function drawSiblingPatches(detailAlpha, colorProgress) {
      if (siblingReveal <= 0.001) return;
      for (let index = 1; index < 4; index += 1) {
        const centerX = patchCenters[index * 2];
        const centerY = patchCenters[index * 2 + 1];
        const rawReveal = clamp(siblingReveal * 1.45 - (index - 1) * 0.16, 0, 1);
        const stagger = easeOutCubic(rawReveal);
        const popScale = lerp(0.58, 1, easeOutBack(rawReveal));
        const popRotation =
          (1 - rawReveal) * (index % 2 === 0 ? -0.12 : 0.12);
        context.save();
        context.translate(centerX, centerY);
        context.rotate(popRotation);
        context.scale(popScale, popScale);
        context.translate(-centerX, -centerY);
        drawPatchSurface(centerX, centerY, stagger, colorProgress);
        const cache = patchCaches[index - 1];
        if (cache) {
          context.save();
          context.globalAlpha = stagger * 0.92 * detailAlpha;
          context.drawImage(
            cache,
            centerX - patchSize * 0.5,
            centerY - patchSize * 0.5,
            patchSize,
            patchSize
          );
          context.restore();
        }
        context.restore();
      }
    }

    function renderScene(interpolation) {
      renderedFrameCount += 1;
      const renderedCameraScale = lerp(
        camera.previousScale,
        camera.scale,
        interpolation
      );
      const renderedCameraX = lerp(camera.previousX, camera.x, interpolation);
      const renderedCameraY = lerp(camera.previousY, camera.y, interpolation);
      const renderedLogoRotation = lerp(
        previousLogoRotation,
        logoRotation,
        interpolation
      );
      const renderedLogoColorProgress = lerp(
        previousLogoColorProgress,
        logoColorProgress,
        interpolation
      );
      const renderedLogoDetailAlpha = lerp(
        previousLogoDetailAlpha,
        logoDetailAlpha,
        interpolation
      );
      const renderedLoopDiveScale = lerp(
        previousLoopDiveScale,
        loopDiveScale,
        interpolation
      );
      const renderedLoopBlackAlpha = lerp(
        previousLoopBlackAlpha,
        loopBlackAlpha,
        interpolation
      );
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      context.clearRect(0, 0, width, height);
      context.save();
      context.translate(width * 0.5 + renderedCameraX, height * 0.5 + renderedCameraY);
      context.scale(
        renderedCameraScale * renderedLoopDiveScale,
        renderedCameraScale * renderedLoopDiveScale
      );
      context.rotate(renderedLogoRotation);

      drawSiblingPatches(renderedLogoDetailAlpha, renderedLogoColorProgress);
      drawPatchSurface(
        currentPatchWorldX(),
        currentPatchWorldY(),
        patchReveal,
        renderedLogoColorProgress
      );

      context.save();
      clipCurrentPatch();
      drawLiveField(renderedCameraScale, renderedLogoDetailAlpha);
      drawQueue(renderedCameraScale, renderedLogoDetailAlpha);
      drawFlyingCopy(interpolation, renderedCameraScale, renderedLogoDetailAlpha);
      drawAuditFrame(interpolation, renderedCameraScale, renderedLogoDetailAlpha);
      context.restore();

      context.restore();

      if (renderedLoopBlackAlpha > 0.001) {
        context.save();
        context.globalAlpha = clamp(renderedLoopBlackAlpha, 0, 1);
        context.fillStyle = COLORS.canvas;
        context.fillRect(0, 0, width, height);
        context.restore();
      }
    }

    function settleImmediately() {
      resetStoryData();
      introProgress = 1;
      patchReveal = 1;
      siblingReveal = 1;
      fieldDim = 0.78;
      logoRotation = Math.PI / 4;
      previousLogoRotation = logoRotation;
      logoColorProgress = 1;
      previousLogoColorProgress = 1;
      logoDetailAlpha = 0;
      previousLogoDetailAlpha = 0;
      loopDiveScale = 1;
      previousLoopDiveScale = 1;
      loopBlackAlpha = 0;
      previousLoopBlackAlpha = 0;
      activeNucleusId = -1;
      queue = [];
      for (let index = 0; index < scanPlan.length; index += 1) {
        const plan = scanPlan[index];
        if (!plan.selected) continue;
        const nucleus = nuclei[plan.nucleusId];
        nucleus.selected = true;
        nucleus.queued = true;
        nucleus.markProgress = 1;
        queue.push({ nucleusId: nucleus.id, slot: queue.length });
      }
      resetCamera(finalCamera);
      setState(STATES.SETTLED);
      completed = true;
      canvas.dataset.complete = "true";
      renderScene(1);
      stopLoop(true);
    }

    function frameLoop(timestamp) {
      if (!running || destroyed) return;
      if (previousFrameTimestamp === 0) previousFrameTimestamp = timestamp;
      const rawDeltaSeconds = Math.max(0, (timestamp - previousFrameTimestamp) / 1000);
      const deltaSeconds = Math.min(rawDeltaSeconds, CONFIG.maxFrameDeltaSeconds);
      previousFrameTimestamp = timestamp;
      accumulator += deltaSeconds;
      renderAccumulator += deltaSeconds;

      while (accumulator >= CONFIG.fixedStepSeconds) {
        snapshotInterpolants();
        updateSimulation(CONFIG.fixedStepSeconds);
        updateCamera(CONFIG.fixedStepSeconds);
        accumulator -= CONFIG.fixedStepSeconds;
      }

      if (renderAccumulator + 0.0005 < CONFIG.renderIntervalSeconds) {
        animationFrame = window.requestAnimationFrame(frameLoop);
        return;
      }
      renderAccumulator = Math.max(
        0,
        renderAccumulator - CONFIG.renderIntervalSeconds
      );
      renderScene(accumulator / CONFIG.fixedStepSeconds);
      if (completed) {
        stopLoop(false);
        return;
      }
      animationFrame = window.requestAnimationFrame(frameLoop);
    }

    function startLoop() {
      if (
        running ||
        destroyed ||
        reducedMotion ||
        !assetsLoaded ||
        !intersecting ||
        !pageVisible ||
        completed
      ) {
        return;
      }
      running = true;
      previousFrameTimestamp = 0;
      accumulator = 0;
      renderAccumulator = 0;
      animationFrame = window.requestAnimationFrame(frameLoop);
    }

    function stopLoop(resetTiming) {
      running = false;
      if (animationFrame) {
        window.cancelAnimationFrame(animationFrame);
        animationFrame = 0;
      }
      if (resetTiming) {
        previousFrameTimestamp = 0;
        accumulator = 0;
        renderAccumulator = 0;
      }
    }

    function handleIntersection(entries) {
      const entry = entries[0];
      intersecting = Boolean(entry && entry.intersectionRatio >= 0.1);
      if (intersecting) startLoop();
      else stopLoop(true);
    }

    function handleVisibilityChange() {
      pageVisible = !document.hidden;
      if (pageVisible) startLoop();
      else stopLoop(true);
    }

    function handleMotionChange(event) {
      reducedMotion = event.matches;
      if (!assetsLoaded) return;
      if (reducedMotion) settleImmediately();
      else {
        rebuildScene();
        startIntro();
        startLoop();
      }
    }

    function handleResize() {
      if (resizeFrame) window.cancelAnimationFrame(resizeFrame);
      resizeFrame = window.requestAnimationFrame(() => {
        resizeFrame = 0;
        if (destroyed) return;
        const wasCompleted = completed;
        measure();
        if (!assetsLoaded) {
          renderScene(1);
          return;
        }
        rebuildScene();
        if (reducedMotion || wasCompleted) settleImmediately();
        else {
          startIntro();
          startLoop();
        }
      });
    }

    const resizeObserver = new ResizeObserver(handleResize);
    const intersectionObserver = new IntersectionObserver(handleIntersection, {
      threshold: [0, 0.1, 0.25],
    });

    function debugSnapshot() {
      let selectedCount = 0;
      for (let index = 0; index < nuclei.length; index += 1) {
        if (nuclei[index].selected) selectedCount += 1;
      }
      let readyAssetCount = 0;
      for (let index = 0; index < sprites.length; index += 1) {
        if (sprites[index].ready) readyAssetCount += 1;
      }
      return {
        state,
        complete: completed,
        reducedMotion,
        mobile: isMobile,
        storyElapsedSeconds: storyElapsed,
        introProgress,
        nucleusCount: nuclei.length,
        selectedCount,
        queueCount: queue.length,
        readyAssetCount,
        failedAssetCount: sprites.length - readyAssetCount,
        qualityTier,
        renderDpr: dpr,
        renderedFrameCount,
        cameraScale: camera.scale,
        patchReveal,
        siblingReveal,
        logoRotationRadians: logoRotation,
        logoColorProgress,
        logoDetailAlpha,
        loopDiveScale,
        loopBlackAlpha,
        cycleCount,
      };
    }

    const debugApi = Object.freeze({ snapshot: debugSnapshot });
    window.__AANCA_HERO__ = debugApi;

    async function initialise() {
      measure();
      resetCamera(initialCamera);
      canvas.dataset.renderer = "canvas2d-png-sprites";
      canvas.dataset.complete = "false";
      setState(STATES.PRELOAD);
      renderScene(1);
      resizeObserver.observe(hero || canvas);
      intersectionObserver.observe(hero || canvas);
      document.addEventListener("visibilitychange", handleVisibilityChange);
      motionQuery.addEventListener("change", handleMotionChange);

      const loadedSprites = await Promise.all(ASSET_DEFINITIONS.map(loadSprite));
      if (destroyed) {
        for (let index = 0; index < loadedSprites.length; index += 1) {
          const sprite = loadedSprites[index];
          if (sprite.closeable && sprite.source && typeof sprite.source.close === "function") {
            sprite.source.close();
          }
        }
        return;
      }

      sprites = loadedSprites;
      assetsLoaded = true;
      let readyCount = 0;
      for (let index = 0; index < sprites.length; index += 1) {
        if (sprites[index].ready) readyCount += 1;
      }
      canvas.dataset.assetsReady = String(readyCount);
      canvas.dataset.assetFailures = String(sprites.length - readyCount);
      measure();
      rebuildScene();
      if (reducedMotion) settleImmediately();
      else {
        startIntro();
        startLoop();
      }
    }

    function destroy() {
      if (destroyed) return;
      destroyed = true;
      stopLoop(true);
      if (resizeFrame) window.cancelAnimationFrame(resizeFrame);
      resizeObserver.disconnect();
      intersectionObserver.disconnect();
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      motionQuery.removeEventListener("change", handleMotionChange);
      for (let index = 0; index < sprites.length; index += 1) {
        const sprite = sprites[index];
        if (sprite.closeable && sprite.source && typeof sprite.source.close === "function") {
          sprite.source.close();
        }
      }
      if (window.__AANCA_HERO__ === debugApi) delete window.__AANCA_HERO__;
    }

    return { initialise, destroy };
  }

  function boot() {
    const canvas = document.querySelector("canvas.hero-canvas");
    if (!canvas) return;
    const controller = createController(canvas);
    if (!controller) return;
    controller.initialise();
    window.addEventListener("pagehide", controller.destroy, { once: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    boot();
  }
})();
