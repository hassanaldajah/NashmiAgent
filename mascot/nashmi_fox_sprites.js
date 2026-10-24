// Nashmi pixel art. One character per pixel:
//   . transparent   K black #000   W white #FFF   L light grey   G mid grey
// FRONT, LEFT and BACK were sampled cell by cell from the reference sheet
// (4 px art grid) and cleaned by hand; RIGHT is LEFT mirrored, as in the
// reference. Animation frames are made at runtime by moving the "parts"
// below or editing single pixels, never by scaling or rotating.
const NASHMI_FOX_SPRITES = {
  views: {
    front: {
      rows: [
      "..KKKK.............KKKK....",
      "..KWWKK...........KKWWK....",
      ".KWWWWWK.........KWWWWWK...",
      ".KWLLWWWK.......KWWWLLWK...",
      ".KWLLLWWK.KKKKK.KWWLLLWK...",
      ".KWLLLWKKKWWWWWKKKWLLLWK...",
      ".KWLLKKWWWWWWWWWWWKKLLWK...",
      ".KWLKWWWWWWWWWWWWWWWKLWK...",
      "..KWWWWWWWWWWWWWWWWWWWK....",
      "..KWWWWWWWWWWWWWWWWWWWK....",
      "..KWWWWWWWWWWWWWWWWWWWK....",
      ".KWWWWWWWWWWWWWWWWWWWWWK...",
      ".KWWWWWWKKWWWWWKKWWWWWWK...",
      "KLWWWWWWKKWWWWWKKWWWWWWLK..",
      "KLWWWWWWKKWWWWWKKWWWWWWLK..",
      ".KLWWWWWWWWWWWWWWWWWWWLK...",
      "..KWWKKWWWWKKKWWWWKKWWK....",
      "...KWWWWWWWWKWWWWWWWWK.....",
      "...KWKKWWWWWWWWWWWKKWK.....",
      "...KWKLLLLLLLLLLLLLKWK.....",
      "....KKKKKKKKKKKKKKKKK..KK..",
      "........KWWWWWWWK.....KKKK.",
      ".......KWWWWWWWWWK....KKKKK",
      ".......KWWWWWWWWWK....KKKKK",
      ".....KKKWWWWWWWWWKKK..KWWWK",
      "....KWWKWWWWWWWWWKWWKKWWWWK",
      "....KWWKWWWLLLWWWKWWKKWWWLK",
      "...KWWWKWWWLKLWWWKWWWKWWWLK",
      "...KWWWKWWWWKWWWWKWWWKWWWK.",
      "...KLWWKWWWWKWWWWKWWLKWWLK.",
      "...KLWWKWWWWKWWWWKWWLKWLK..",
      "...KLLWKWWWWKWWWWKWLLKLK...",
      "...KWWWKWWWWKWWWWKWWWKK....",
      "...KWKWKWWKWKWKWWKWKWK.....",
      "....KKKKKKKKKKKKKKKKK......"
    ],
      eyes: [{ x: 8, y: 12, w: 2, h: 3 }, { x: 15, y: 12, w: 2, h: 3 }],
      // rows from here down stay put when he breathes in (head rises 1 px)
      seam: 22,
      parts: {
        earR: { x: 18, y: 0, w: 7, h: 5 },
        tailTip: { x: 22, y: 20, w: 5, h: 6 },
        tailBase: { x: 22, y: 26, w: 5, h: 7 },
        // his right front leg is on the viewer's left
        pawR: { x: 8, y: 25, w: 4, h: 10, belly: true },
        pawL: { x: 13, y: 25, w: 4, h: 10, belly: true }
      }
    },
    left: {
      rows: [
      "......KKKK..............",
      "......KWWKK.............",
      "......KWWWKK............",
      "......KWLWWK............",
      "......KWLLWWK...........",
      "......KWLLWWK.KK........",
      "....KKKWLLLWKKWLK.......",
      "...KWWWKLLLWWKWLK.......",
      "..KWWWWWKLLWWWWWWK......",
      ".KWWWWWWWWWWWWWWWK......",
      ".KWWWWWWWWWWWWWWWK......",
      ".KWWWWWWWWWWWWWWWK......",
      ".KWWWWWWWWWWWWWWWK......",
      ".KWWKWWWWWWWWWWWWK......",
      ".KWWKWWWWWWWWWWLLK......",
      ".KWWKWWWWWWWWLLLLK......",
      "KWWWWWWWWWWLLLLLLK......",
      "KKWWWKKWWWWLLLLLLK......",
      ".KWWWWWWWWLLLLLLK.......",
      "..KWWKWWWLLLLLLK..KKK...",
      "...KLLLLLLLLLLK..KKKKK..",
      "....KKKKKKKKKKK.KKKKKKK.",
      "......KWWWWWWWWK.KKKKKKK",
      ".....KWWWWWWWWWWK.KKKKKK",
      ".....KWWWWWWWWWWK.KWWWWK",
      ".....KWWWWWWWWWWKKWWWWWK",
      "....KWWWWWWWWWWWKKWWWWLK",
      "....KWWWWWWWWWWWKKWWWLK.",
      "....KWWWKWWWWWWWKKWWWLK.",
      "....KWWWKWWWWWWLKKWWLK..",
      "....KWWLKWWWWWWLKKWLLK..",
      "...KWWWLKWWWWWLLKKWLK...",
      "...KWWLKWWWWWLLLKKLK....",
      "...KWLLKLWWWLLLLKKK.....",
      "...KLLKKLLLLLLLKK.......",
      "...KKKKKKKKKKKKKK......."
    ],
      eyes: [{ x: 4, y: 13, w: 1, h: 3 }],
      seam: 23,
      parts: {
        tailTip: { x: 17, y: 19, w: 7, h: 6 },
        tailBase: { x: 17, y: 25, w: 7, h: 9 },
        legF: { x: 3, y: 31, w: 5, h: 5 },
        legH: { x: 8, y: 31, w: 9, h: 5 }
      }
    },
    back: {
      rows: [
      "...KKKK.............KKKK...",
      "...KWWWK...........KWWWK...",
      "...KWLWWK.........KWWLWK...",
      "...KWLLWWK.......KWWLLWK...",
      "...KWLLLWWK.....KWWLLLWK...",
      "...KWLLLLWKKKKKKKWLLLLWK...",
      "...KWLLLLKWWWWWWWKLLLLWK...",
      "...KWLLLKWWWWWWWWWKLLLWK...",
      "...KKKKKWWWWWWWWWWWKKKKK...",
      "....KWWWWWWWWWWWWWWWWWK....",
      "...KWWWWWWWWWWWWWWWWWWWK...",
      "...KWWWWWWWWWWWWWWWWWWWK...",
      "..KWWWWWWWWWWWWWWWWWWWWWK..",
      "..KWWWWWWWWWWWWWWWWWWWWWK..",
      "..KWWWWWWWWWWWWWWWWWWWWWK..",
      "...KWWWWWWWWWWWWWWWWWWWK...",
      "...KLWWWWWWWWWWWWWWWWWLK...",
      "....KLLWWWWWWWWWWWWWLLK....",
      "....KLLLLLLLLLLLLLLLLLK....",
      ".....KKLLLLLLLLLLLLLKK.....",
      "..KKK..KKKKKKKKKKKKK.......",
      ".KKKKK..KWWWWWWWWWK........",
      ".KKKKK..KWWWWWWWWWK........",
      "KKKKKKK.KWWWWWWWWWWK.......",
      "KKKKKKK.KWWWWWWWWWWK.......",
      "KWWWWWK.KWWWWWWWWWWWK......",
      "KWWWWLK.KWWWWWWWWWWWK......",
      "KLWWWLKKWWWWWWWWWWWWK......",
      "KLWWWWKKWWWWWWWWWWWWWK.....",
      "KLWWWWLKWWWWWWWWWWWWWK.....",
      ".KLWWWWKKKKKKKWWWWWWWK.....",
      ".KLWWWWWWWWWWWKKWWWWWLK....",
      "..KLWWWWWWWWWLKWWWWWLK.....",
      "..KLLWWWWWWLLLKLLWWLLK.....",
      "...KLLLLLLLLLKKLLLLLKK.....",
      "...KKKKKKKKKKKKKKKKKK......"
    ],
      eyes: [],
      seam: 22,
      parts: {
        tailTip: { x: 0, y: 20, w: 7, h: 5 },
        tailBase: { x: 0, y: 25, w: 7, h: 5 }
      }
    }
  },
  // Side-view laptop (the fox sits sideways at it, like Claude's crab).
  // Frames go closed -> open, all 14 rows tall and sharing the bottom row.
  // "S" marks screen pixels; they face the fox and flicker while he types.
  laptop: {
    frames: [
      [
        "................",
        "................",
        "................",
        "................",
        "................",
        "................",
        "................",
        "................",
        "................",
        "................",
        "...KKKKKKKKKKKKK",
        "...KGGGGGGGGGGGK",
        "...KLGLGLGLGLGLK",
        "...KKKKKKKKKKKKK"
      ],
      [
        "................",
        "................",
        "................",
        "................",
        "................",
        "................",
        ".............KK.",
        "...........KGGK.",
        ".........KGGK...",
        ".......KGGK.....",
        ".....KGGK.......",
        "...KGGK.........",
        "...KLGLGLGLGLGLK",
        "...KKKKKKKKKKKKK"
      ],
      [
        "...KKKK.........",
        "...KGSK.........",
        "...KGSK.........",
        "...KGSK.........",
        "...KGSK.........",
        "...KGSK.........",
        "...KGSK.........",
        "...KGSK.........",
        "...KGSK.........",
        "...KGSK.........",
        "...KGSK.........",
        "...KGSK.........",
        "...KLGLGLGLGLGLK",
        "...KKKKKKKKKKKKK"
      ],
      [
        "KKK.............",
        "KGSK............",
        "KGSK............",
        ".KGSK...........",
        ".KGSK...........",
        ".KGSK...........",
        "..KGSK..........",
        "..KGSK..........",
        "..KGSK..........",
        "...KGSK.........",
        "...KGSK.........",
        "...KGSK.........",
        "...KLGLGLGLGLGLK",
        "...KKKKKKKKKKKKK"
      ]
    ],
    // the LEFT view column the laptop's right end reaches (under his paw)
    tuck: 5
  },
  burst: [
    ["..W..", ".WKW.", "WK.KW", ".WKW.", "..W.."],
    ["W...W", ".K.K.", ".....", ".K.K.", "W...W"]
  ]
};
