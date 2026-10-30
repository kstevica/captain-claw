# Captain Bot

The Bot character from the device clients (`firmware/device/src/face.cpp`,
`drawBot`), restyled six ways so it can sit next to the Captain brand: navy,
gold and cyan from the app icon, coral from the website. Every drawing keeps
the Bot's eyes, grille mouth and antenna lamp; the maritime part is what
changes. Everything is a rounded rect, circle, arc, line or triangle, so each
one ports to the M5Canvas calls in the firmware without image files.

| File | What it is | Used by |
| --- | --- | --- |
| `captains-cap.svg` | Navy cap with a gold band, a tiny helm as the badge | **the app icon** (`build_support/icon.svg` is this on the navy slab) |
| `at-the-helm.svg` | The ship's wheel behind the head, the lamp as the top knob | later |
| `sailor.svg` | Dixie-cup hat, navy collar, coral neckerchief, Breton stripes | later |
| `deep-sea-diver.svg` | Brass diving helmet: porthole plate, side ports, hoses | later |
| `beacon-buoy.svg` | The head on a red-and-white buoy on two waves, the lamp a beacon | later |
| `anchor.svg` | Navy body, gold anchor: stock across the head, flukes below | later |

Each has an `-ink.svg` twin: the same geometry in 1-bit for the e-paper
clients, where fills become outlines and colour becomes ink, as the Waveshare
sleeper is drawn. `sheet.html` shows all six in colour on the round screen and
in ink on e-paper; open it from disk.

The drawings are in the firmware's own units (the head is 88 x 76, centred on
0,0) under a `transform` that frames them in a 240 x 240 box, so the numbers
can be read straight into `fillRoundRect` calls.

The icon is rendered from `build_support/icon.svg` by `build_support/render_icon.py`,
which writes every icon file in the repo (and the website's with `--website DIR`).
