// Nashmi engine. Runs in the main Streamlit page (nashmi_mascot.py injects
// it once) and needs NASHMI_FOX_CONFIG, NASHMI_FOX_SPRITES and
// NASHMI_FOX_VERSION in scope. Everything is drawn on tiny canvases at art
// resolution and scaled by whole device pixels, so the pixels stay square.
//
// The idle loop is modelled on Claude's crab: he sits sideways at a laptop
// seen from the side and taps the keys at an uneven pace; when the pointer
// comes close he shuts the lid, turns to face it and follows it.
(() => {
  'use strict';
  const win = window, doc = document;
  const old = win.__nashmiFox;
  if (old) {
    if (old.version === NASHMI_FOX_VERSION) return;
    old.destroy();
  }

  const C = NASHMI_FOX_CONFIG, S = NASHMI_FOX_SPRITES;
  const PAL = { K: '#000000', W: '#FFFFFF', L: '#D0D0D0', G: '#9A9A9A' };
  const AW = 48, AH = 44;              // art canvas: fox + side laptop + room to hop/swat
  const FOX_H = 35;                    // front sprite height, used for sizing
  const HIT_DX = -8;                   // swat landing point vs. canvas centre (art px)
  const BAR = '[data-testid="stChatInput"]';
  const BUSY = '[data-test-script-state="running"]';
  const rnd = (a, b) => a + Math.random() * (b - a);
  const pick = ([a, b]) => rnd(a, b);

  // ---- sprites -----------------------------------------------------------
  const prep = v => ({ rows: v.rows, w: v.rows[0].length, h: v.rows.length,
                       eyes: v.eyes, seam: v.seam, parts: v.parts });
  function mirror(v) {
    const mx = r => Object.assign({}, r, { x: v.w - r.x - r.w });
    const parts = {};
    for (const k in v.parts) parts[k] = mx(v.parts[k]);
    return { rows: v.rows.map(r => r.split('').reverse().join('')), w: v.w, h: v.h,
             seam: v.seam, eyes: v.eyes.map(mx), parts, mirrored: true };
  }
  const V = { front: prep(S.views.front), left: prep(S.views.left), back: prep(S.views.back) };
  V.right = mirror(V.left);
  const LAP_W = S.laptop.frames[0][0].length, LAP_H = S.laptop.frames[0].length;
  const LAP_X = Math.floor((AW - V.left.w) / 2) + S.laptop.tuck - LAP_W + 1;
  const SCREEN = S.laptop.frames[3].join('').split('S').length - 1;

  // Leg offsets for the 4-frame walk, written for the LEFT view.
  const WALK = [
    { parts: { legF: { dx: -1 }, legH: { dx: 1 } }, rise: 0 },
    { parts: { legF: { dy: -1 } }, rise: 1 },
    { parts: { legF: { dx: 1 }, legH: { dx: -1 } }, rise: 0 },
    { parts: { legH: { dy: -1 } }, rise: 1 },
  ];
  const TAIL = [0, 1, 0, -1];          // 3 distinct wag frames, ping-pong

  // Eyes: shifted `look` px toward the pointer; a blink leaves a 1 px line.
  function eyeCell(v, x, y, c, look, blink) {
    for (const e of v.eyes) {
      if (y < e.y || y >= e.y + e.h) continue;
      const nx = e.x + look, w = blink ? Math.max(e.w, 2) : e.w;
      const bx = v.mirrored ? nx + e.w - w : nx;
      if (x >= bx && x < bx + w) return !blink || y === e.y + (e.h >> 1) ? 'K' : 'W';
      if (x >= e.x && x < e.x + e.w) return 'W';
    }
    return c;
  }

  function put(ctx, x, y, c) {
    if (!c || c === '.') return;
    ctx.fillStyle = PAL[c];
    ctx.fillRect(x, y, 1, 1);
  }

  function drawView(ctx, name, pose, ox, oy) {
    const v = V[name], parts = v.parts, P = pose.parts || {};
    const sign = v.mirrored ? -1 : 1, rise = pose.rise || 0;
    const lift = y => (y < v.seam ? -rise : 0);
    const eyed = v.eyes.length && (pose.blink || pose.look);
    const cell = (x, y) => {
      const c = v.rows[y][x];
      return eyed ? eyeCell(v, x, y, c, pose.look || 0, pose.blink) : c;
    };
    // parts posed with `keep` also stay drawn in place (the far paw of a pair)
    const partAt = (x, y) => {
      for (const k in parts) {
        const r = parts[k];
        if (!(P[k] && P[k].keep) && x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h) return r;
      }
      return null;
    };
    // body without its movable parts (legs leave belly behind, the rest leaves a gap)
    for (let y = 0; y < v.h; y++) {
      for (let x = 0; x < v.w; x++) {
        const r = partAt(x, y);
        let c = cell(x, y);
        if (r) {
          if (!r.belly) continue;
          c = y === r.y + r.h - 1 ? 'K' : 'W';
        }
        put(ctx, ox + x, oy + y + lift(y), c);
      }
    }
    if (rise) {                        // breathing in: the seam row stretches by 1 px
      for (let x = 0; x < v.w; x++) {
        if (!partAt(x, v.seam)) put(ctx, ox + x, oy + v.seam - 1, cell(x, v.seam));
      }
    }
    for (const k in parts) {
      const r = parts[k], p = P[k] || {};
      // a part rises only if it sits wholly above the seam (no gap in the tail)
      const dx = (p.dx || 0) * sign, dy = (p.dy || 0) + (r.y + r.h <= v.seam ? -rise : 0);
      const solid = (x, y) => x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h
                              && cell(x, y) !== '.';
      if (p.outline) {
        for (let y = r.y; y < r.y + r.h; y++) {
          for (let x = r.x; x < r.x + r.w; x++) {
            if (!solid(x, y)) continue;
            for (const [nx, ny] of [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]) {
              if (!solid(nx, ny)) put(ctx, ox + nx + dx, oy + ny + dy, 'K');
            }
          }
        }
      }
      for (let y = r.y; y < r.y + r.h; y++) {
        for (let x = r.x; x < r.x + r.w; x++) put(ctx, ox + x + dx, oy + y + dy, cell(x, y));
      }
    }
  }

  function drawRows(ctx, rows, ox, oy) {
    rows.forEach((row, y) => { for (let x = 0; x < row.length; x++) put(ctx, ox + x, oy + y, row[x]); });
  }

  // Side laptop; `lit` screen pixels glow white, top to bottom, the rest stay grey.
  function drawLaptop(ctx, lid, lit, mirrored) {
    const rows = S.laptop.frames[lid], oy = AH - LAP_H;
    let n = 0;
    rows.forEach((row, y) => {
      for (let x = 0; x < LAP_W; x++) {
        let c = row[x];
        if (c === 'S') c = n++ < lit ? 'W' : 'L';
        put(ctx, mirrored ? AW - 1 - (LAP_X + x) : LAP_X + x, oy + y, c);
      }
    });
  }

  // ---- DOM -----------------------------------------------------------------
  const layer = doc.createElement('div');
  layer.id = 'nashmi-fox-layer';
  layer.setAttribute('aria-hidden', 'true');
  layer.style.cssText = 'position:fixed;left:0;top:0;width:0;height:0;'
    + 'pointer-events:none;z-index:' + C.zIndex + ';';
  function canvas(w, h) {
    const c = doc.createElement('canvas');
    c.width = w; c.height = h;
    c.style.cssText = 'position:fixed;left:0;top:0;display:none;pointer-events:none;'
      + 'image-rendering:crisp-edges;image-rendering:pixelated;';
    layer.appendChild(c);
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingEnabled = false;
    return { c, ctx, key: '' };
  }
  const lap = canvas(AW, AH), fox = canvas(AW, AH), burst = canvas(5, 5);
  doc.body.appendChild(layer);

  // ---- state ---------------------------------------------------------------
  const mq = q => (win.matchMedia ? win.matchMedia(q) : { matches: false });
  const mqReduced = mq('(prefers-reduced-motion: reduce)');
  const mqHover = mq('(hover: hover) and (pointer: fine)');
  let reduced = mqReduced.matches, canHover = mqHover.matches;
  let bar = null, radius = 0, rtl = false, busy = false, lastScan = -1e9, lastBusy = -1e9;
  let ptr = null, lastUserType = -1e9, raf = 0, last = performance.now();
  const t0 = performance.now();
  const st = {
    mode: 'home', view: 'left', x: null,
    lid: 3, lidT: 0, modeT: t0,
    walk: 0, walkT: 0, tail: 0, tailT: 0,
    breath: 0, breathT: t0,
    nextBlink: t0 + pick(C.blinkEvery), blinkUntil: 0,
    nextEar: t0 + pick(C.earTwitchEvery), earUntil: 0,
    nextKey: t0, tap: 0, lit: 0, fullAt: 0,
    swat: null, swatT: 0, nextSwat: 0,
    hopI: -1, hopT: 0, shakeI: -1, shakeT: 0, burstAt: -1e9, burstX: 0, burstY: 0,
  };

  // ---- geometry --------------------------------------------------------------
  function unit() {                    // CSS px per art px, whole device pixels
    const dpr = win.devicePixelRatio || 1;
    if (typeof C.scale === 'number') return C.scale;
    return Math.max(1, Math.round(C.targetHeight * dpr / FOX_H)) / dpr;
  }
  const snap = v => { const d = win.devicePixelRatio || 1; return Math.round(v * d) / d; };
  function scanBar(now) {
    lastScan = now;
    if (bar && bar.isConnected) return;
    bar = doc.querySelector(BAR);
    if (!bar) return;
    const cs = win.getComputedStyle(bar);
    radius = parseFloat(cs.borderTopLeftRadius) || 0;
    rtl = cs.direction === 'rtl';
  }
  function inZone(p, r) {
    if (!p || p.x < r.left || p.x > r.right) return false;
    const rad = Math.min(radius, r.width / 2, r.height / 2);
    const d = Math.max(r.left + rad - p.x, p.x - (r.right - rad));
    const edge = rad > 0 && d > 0 ? r.top + rad - Math.sqrt(Math.max(0, rad * rad - d * d)) : r.top;
    return p.y >= edge - C.zoneAbove && p.y <= edge + C.zoneBelow;
  }
  function place(cv, x, r, u, dx) {
    cv.c.style.width = AW * u + 'px';
    cv.c.style.height = AH * u + 'px';
    cv.c.style.left = snap(x - AW * u / 2 + (dx || 0)) + 'px';
    cv.c.style.top = snap(r.top - AH * u) + 'px';
    cv.c.style.display = 'block';
  }
  function hideAll() { for (const cv of [lap, fox, burst]) cv.c.style.display = 'none'; }

  function render(cv, key, draw) {
    if (cv.key === key) return;
    cv.key = key;
    cv.ctx.clearRect(0, 0, cv.c.width, cv.c.height);
    draw(cv.ctx);
  }

  // ---- main loop ---------------------------------------------------------------
  function tick(now, dt) {
    if (now - lastScan > 500) scanBar(now);
    if (now - lastBusy > 250) { lastBusy = now; busy = !!doc.querySelector(BUSY); }
    const r = bar && bar.isConnected ? bar.getBoundingClientRect() : null;
    if (!r || r.width < 60 || r.bottom < 0 || r.top > win.innerHeight) { hideAll(); return; }

    const u = unit();
    const right = C.homeCorner === 'right' || (C.homeCorner !== 'left' && !rtl);
    const idleView = right ? 'left' : 'right';   // he faces into the bar, laptop in front
    let lo = r.left + C.homeInset + 14 * u, hi = r.right - C.homeInset - 13 * u;
    if (lo > hi) lo = hi = (lo + hi) / 2;
    const home = right ? hi : lo;
    if (st.x === null) st.x = home;
    st.x = Math.min(hi, Math.max(lo, st.x));
    const zone = canHover && !reduced && inZone(ptr, r);
    const target = ptr ? Math.min(hi, Math.max(lo, ptr.x - HIT_DX * u)) : home;
    if (st.mode === 'home' || st.mode === 'open' || st.mode === 'close') st.x = home;

    const go = m => { st.mode = m; st.modeT = now; };
    const walkTo = (to, speed) => {
      const d = to - st.x, step = speed * dt;
      st.view = d < 0 ? 'left' : 'right';
      if (Math.abs(d) <= step) { st.x = to; return true; }
      st.x += Math.sign(d) * step;
      if (now - st.walkT >= C.walkFrameMs) { st.walkT = now; st.walk = (st.walk + 1) % 4; }
      return false;
    };
    const startSwat = () => { go('swat'); st.swat = 'wind'; st.swatT = now; st.view = 'front'; };
    const lidStep = dir => {
      if (now - st.lidT < C.lidFrameMs) return false;
      st.lidT = now;
      st.lid = Math.max(0, Math.min(3, st.lid + dir));
      return dir < 0 ? st.lid === 0 : st.lid === 3;
    };
    const leave = () => {
      if (C.chase) { go('annoyed'); st.view = 'back'; } else go('settle');
    };

    switch (st.mode) {
      case 'home':
        st.view = idleView;
        if (zone) { go('close'); st.lidT = 0; }
        break;
      case 'close':                    // shut the lid, then turn to face the pointer
        st.view = idleView;
        if (!zone) go('open');
        else if (lidStep(-1)) { go('face'); st.view = 'front'; st.hopI = 0; st.hopT = now; }
        break;
      case 'face':
        st.view = 'front';
        if (!zone) leave();
        else if (now - st.modeT >= C.frameMs) {
          if (C.chase) go('chase'); else go('watch');
        }
        break;
      case 'open':
        st.view = idleView;
        if (zone) go('close');
        else if (lidStep(1)) { go('home'); st.nextKey = now; }
        break;
      case 'chase':
        if (!zone) { leave(); break; }
        if (walkTo(target, C.chaseSpeed)) startSwat();
        break;
      case 'swat': {
        st.view = 'front';
        const t = now - st.swatT;
        if (st.swat === 'wind' && t >= C.frameMs / 2) { st.swat = 'up'; st.swatT = now; }
        else if (st.swat === 'up' && t >= C.swatUpMs) {
          st.swat = 'hit'; st.swatT = now; st.shakeI = 0; st.shakeT = now;
          st.burstAt = now; st.burstX = ptr ? ptr.x : st.x + HIT_DX * u;
          st.burstY = Math.min(ptr ? ptr.y : r.top, r.top);
        } else if (st.swat === 'hit' && t >= C.swatHitMs) {
          go('watch'); st.swat = null; st.nextSwat = now + pick(C.swatInterval);
        }
        break;
      }
      case 'watch':
        st.view = 'front';
        if (!zone) leave();
        else if (!C.chase) break;
        else if (Math.abs(st.x + HIT_DX * u - ptr.x) > C.swatReach && Math.abs(target - st.x) > 1) go('chase');
        else if (now >= st.nextSwat) startSwat();
        break;
      case 'annoyed':
        st.view = 'back';
        if (zone) go('chase');
        else if (now - st.modeT >= C.annoyedMs) go('return');
        break;
      case 'return':
        if (zone) { go('chase'); break; }
        if (walkTo(home, C.returnSpeed)) go('settle');
        break;
      case 'settle':                   // a little stretch, then sit back down at the laptop
        st.view = 'front';
        if (zone) { go(C.chase ? 'chase' : 'watch'); break; }
        if (now - st.modeT >= C.settleMs) { go('open'); st.view = idleView; st.lidT = now; }
        break;
    }
    if (reduced && st.mode !== 'home') { go('home'); st.lid = 3; st.x = home; st.view = idleView; }

    // idle timers
    const tailMs = reduced ? C.tailMs.reduced : busy ? C.tailMs.busy : C.tailMs.normal;
    if (now - st.tailT >= tailMs) { st.tailT = now; st.tail = (st.tail + 1) % TAIL.length; }
    if (now >= st.nextBlink) { st.blinkUntil = now + C.blinkMs; st.nextBlink = now + pick(C.blinkEvery); }
    if (!reduced && now >= st.nextEar) { st.earUntil = now + C.earTwitchMs; st.nextEar = now + pick(C.earTwitchEvery); }
    if (!reduced && now - st.breathT >= C.breathMs) { st.breathT = now; st.breath ^= 1; }
    if (st.mode === 'home' && !reduced && now >= st.nextKey) {
      const userTyping = now - lastUserType < C.userTypingHoldMs;
      const ms = busy ? C.typeMs.busy : userTyping ? C.typeMs.userTyping : C.typeMs.normal;
      st.tap ^= 1;
      // a full screen stays up for textClearMs while he keeps typing, then clears
      if (st.fullAt) { if (now - st.fullAt >= C.textClearMs) { st.lit = 0; st.fullAt = 0; } }
      else if (++st.lit >= SCREEN) st.fullAt = now;
      st.nextKey = now + ms * rnd(1 - C.typeJitter, 1 + C.typeJitter)
        + (!busy && !userTyping && Math.random() < C.pauseChance ? pick(C.pauseMs) : 0);
    }
    let hop = 0;
    if (st.hopI >= 0) {
      const i = Math.floor((now - st.hopT) / C.hopStepMs);
      if (i >= C.hop.length) st.hopI = -1; else hop = C.hop[i];
    }
    let shake = 0;
    if (st.shakeI >= 0) {
      const i = Math.floor((now - st.shakeT) / (C.frameMs / 2));
      if (i >= C.shake.length) st.shakeI = -1; else shake = C.shake[i];
    }

    // pose
    const v = V[st.view], moving = st.mode === 'chase' || st.mode === 'return';
    const typing = st.mode === 'home';
    const pose = { parts: { tailTip: { dx: TAIL[st.tail] } }, rise: 0, blink: now < st.blinkUntil, look: 0 };
    if (moving) {
      pose.rise = WALK[st.walk].rise;
      Object.assign(pose.parts, WALK[st.walk].parts);
    } else if (st.mode === 'settle') pose.rise = 1;
    else if (!reduced) pose.rise = st.breath;
    if (typing) pose.parts.legF = { dx: -2, dy: reduced || st.tap ? -2 : -4, keep: true };
    if (st.view === 'front' && now < st.earUntil) pose.parts.earR = { dy: 1 };
    if (st.view === 'front' && C.eyeFollow && ptr) {
      const d = (ptr.x - st.x) / u;
      pose.look = d < -6 ? -1 : d > 6 ? 1 : 0;
    }
    if (st.mode === 'swat') {
      pose.parts.pawR = st.swat === 'wind' ? { dy: -2 }
        : st.swat === 'up' ? { dx: -4, dy: -6, outline: true } : { dx: -4, outline: true };
      if (st.swat === 'hit') pose.parts.pawL = { dy: -1 };
      pose.rise = 0;
    }
    const withLaptop = st.mode === 'home' || st.mode === 'open' || st.mode === 'close';
    const lit = typing ? (reduced ? SCREEN >> 1 : st.lit) : 0;
    const ox = Math.floor((AW - v.w) / 2), oy = AH - v.h + hop;
    const key = [st.view, JSON.stringify(pose), withLaptop, st.lid, lit, hop].join('|');

    render(fox, key, ctx => {
      if (withLaptop) drawLaptop(ctx, st.lid, lit, !right);
      drawView(ctx, st.view, pose, ox, oy);
    });
    place(fox, st.x, r, u, shake);

    if (withLaptop) lap.c.style.display = 'none';
    else {
      render(lap, 'closed' + right, ctx => drawLaptop(ctx, 0, 0, !right));
      place(lap, home, r, u);
    }

    const bt = now - st.burstAt;
    if (bt < C.burstMs) {
      const f = bt < C.burstMs / 2 ? 0 : 1;
      render(burst, 'b' + f, ctx => drawRows(ctx, S.burst[f], 0, 0));
      burst.c.style.width = burst.c.style.height = 5 * u + 'px';
      burst.c.style.left = snap(st.burstX - 2.5 * u) + 'px';
      burst.c.style.top = snap(st.burstY - 2.5 * u) + 'px';
      burst.c.style.display = 'block';
    } else burst.c.style.display = 'none';
  }

  function loop(now) {
    raf = win.requestAnimationFrame(loop);
    const dt = Math.min(0.05, Math.max(0, (now - last) / 1000));
    last = now;
    try { tick(now, dt); } catch (err) { console.error('Nashmi fox:', err); destroy(); }
  }

  // ---- input ----------------------------------------------------------------
  const onMove = e => { if (e.pointerType !== 'touch') ptr = { x: e.clientX, y: e.clientY }; };
  const onOut = e => { if (!e.relatedTarget) ptr = null; };
  const onBlur = () => { ptr = null; };
  const onType = e => { if (bar && e.target && bar.contains(e.target)) lastUserType = performance.now(); };
  const onDown = e => {
    if (reduced || fox.c.style.display === 'none') return;
    const b = fox.c.getBoundingClientRect();
    if (e.clientX < b.left || e.clientX >= b.right || e.clientY < b.top || e.clientY >= b.bottom) return;
    const ax = Math.floor((e.clientX - b.left) / b.width * AW);
    const ay = Math.floor((e.clientY - b.top) / b.height * AH);
    if (fox.ctx.getImageData(ax, ay, 1, 1).data[3] === 0) return;
    st.hopI = 0; st.hopT = performance.now();
  };
  const onReduced = () => { reduced = mqReduced.matches; };
  const onHover = () => { canHover = mqHover.matches; };
  const listen = [
    [doc, 'pointermove', onMove], [doc, 'mouseout', onOut], [win, 'blur', onBlur],
    [doc, 'keydown', onType], [doc, 'input', onType], [doc, 'pointerdown', onDown],
  ];
  for (const [t, n, f] of listen) t.addEventListener(n, f, { capture: true, passive: true });
  const watch = (m, f) => m.addEventListener && m.addEventListener('change', f);
  watch(mqReduced, onReduced);
  watch(mqHover, onHover);

  function destroy() {
    win.cancelAnimationFrame(raf);
    for (const [t, n, f] of listen) t.removeEventListener(n, f, { capture: true });
    if (mqReduced.removeEventListener) mqReduced.removeEventListener('change', onReduced);
    if (mqHover.removeEventListener) mqHover.removeEventListener('change', onHover);
    layer.remove();
    if (win.__nashmiFox === api) delete win.__nashmiFox;
  }
  const api = { version: NASHMI_FOX_VERSION, destroy, state: st };
  win.__nashmiFox = api;
  win.addEventListener('pagehide', destroy, { once: true });
  raf = win.requestAnimationFrame(loop);
})();
