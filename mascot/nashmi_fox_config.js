// Every tunable number for Nashmi, the pixel fox on the composer's top edge.
// Times are in milliseconds, distances in CSS pixels unless noted.
const NASHMI_FOX_CONFIG = {
  // Size. 'auto' picks the whole number of *device* pixels per art pixel
  // that gets the fox closest to targetHeight (crisp on every screen).
  // A number forces that many CSS pixels per art pixel (1, 2, 3...).
  scale: 'auto',
  targetHeight: 56,

  // 'auto' = top-right corner, or top-left when the composer is RTL.
  homeCorner: 'auto',          // 'auto' | 'right' | 'left'
  homeInset: 12,               // gap between his tail/ear and the bar's end

  // true: he runs along the top edge to the pointer and swats it.
  // false: like Claude's crab, he only closes the laptop, turns to face
  // you and follows the pointer with his eyes until it leaves.
  chase: true,
  eyeFollow: true,

  // Trigger zone: a strip along the bar's top border only.
  zoneAbove: 24,
  zoneBelow: 16,

  // Movement (snappy, frame-based, no easing).
  chaseSpeed: 380,             // px/s while running after the pointer
  returnSpeed: 300,            // px/s walking back home
  walkFrameMs: 80,
  frameMs: 90,                 // general sprite frame length

  // Swatting.
  swatReach: 14,               // how close (px) the paw must land to count
  swatInterval: [400, 600],    // repeat swats while the pointer stays
  swatUpMs: 90,
  swatHitMs: 130,
  burstMs: 150,
  shake: [2, -2, 1, 0],        // px per frame on impact

  // Idle at the laptop.
  typeMs: { normal: 90, userTyping: 60, busy: 45 },
  typeJitter: 0.4,             // each keypress is typeMs * (1 +/- jitter)
  pauseChance: 0.05,           // chance per keypress of a short think...
  pauseMs: [300, 700],         // ...this long (never while busy)
  userTypingHoldMs: 1500,
  tailMs: { normal: 120, busy: 70, reduced: 360 },
  blinkEvery: [2000, 5000],
  blinkMs: 110,
  earTwitchEvery: [4000, 8000],
  earTwitchMs: 120,
  breathMs: 650,
  lidFrameMs: 60,
  textClearMs: 400,            // pause before the screen clears when full

  annoyedMs: 300,              // back view when the pointer leaves
  settleMs: 160,               // little stretch at home before sitting down
  hop: [-1, -2, -2, -1, 0],    // art pixels per step on click / tap
  hopStepMs: 40,

  zIndex: 120
};
