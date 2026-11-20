# MERGED RULE (Claude, driver) - plate GO / NO-GO on every layer - 20261120-1135

Sources: ANSWER CHATGPT - 20261120 - CONSENSUS go-nogo every layer (10:26); ANSWER GROK-BUILD ... go-nogo every layer R2 (10:45); ANSWER GROK-BUILD - CONSENSUS band findings in go-nogo (10:40). ChatGPT has not yet answered the band follow-up (MIKE 1037). Mike's orders this rule serves are quoted in MIKE 1020 and MIKE 1037. Gap is 4 mm, locked by Mike 10:53 today (replaces the 3 mm the answers mention).

## 1. Inputs
Layout only. Film, resin, printer are constants and are not compared. Every layer from layer 0 to the last layer. No sampled band, no fitted or borrowed cut-off.

## 2. Hard gate (both AIs agree) - the ONLY things that make a plate NO-GO, and only after the repair and swap steps
- cured cells outside the printable bed (touching the edge is a warning)
- two models in the same cured cell
- a component that never connects, on any layer, to something that reaches the build plate
- a sealed cavity inside a part, or a cup sealed against the film, with no open path
- a sealed pocket between two parts
Raw UVtools counts are not a limit. Each hit is verified against connected geometry.

## 3. The shrinking band (both agree it is computed on every layer; one shared front from the bed edge inward, one grid step at a time)
Per model, per layer, record three events:
- START: a nested model (its first cured cells are not on the plate's outer outline) is met on one open side, instead of shielded on two sides. Rim models meeting the bed edge are the normal outer front and are never flagged.
- LANDING: at the step a larger neighbour's last cell lets go, the smaller model is fully held / already free / touched at one front cell / mid-peel on an open edge. Open edge is the event.
- END: the plate's last loop is on empty bed / on one model / shared by two or more models. Shared is the event.
How the band is used: it is BINDING on the layout. Layouts of the same plate are compared in this order: any hard defect, then a shared last loop, then open-edge landings (fewer is better), then one-sided nested starts. A move, rotation or swap that clears a worse event without creating a hard defect or a new event of the same class is taken. No persistence vote across layers; identical repeated layers count once; the layer interval is reported.
What the band is NOT: a veto. The same events are on plates that printed clean, and plates with none of them failed (best after-the-fact cut 13 to 16 of 19 against 12 of 19 for calling everything failed; two-side protection passed 45 of 45 pieces on clean and failed plates alike). A plate whose hard gate is clean and which has no legal improving move left is a GO with the leftover event written on it (model, layer interval, event).

## 4. Vents and traps
Flood fill each layer: (i) trap inside one part, (ii) cup against the film, (iii) pocket between parts. (i)(ii): vent from the earliest sealed region through the nearest wall to open resin on the plate-facing end, smallest channel the real slice keeps open, widened one pixel step at a time, re-sliced and re-checked on every layer of the sealed interval, on the solver grid and the sliced file. (iii): move, rotate or swap; never drill a model to fix a gap between models. If the vent would cut the part into a new unsupported piece or sealed volume, the model goes to swap.

## 5. Order per plate
Score every layer -> repair vents -> rescore -> rotate / reposition -> rescore -> a hard defect that survives removes the model that owns it -> replacement must fit, pass the hard gate and not worsen the worst event (preference: hard gate, complete set, worst event, fill) -> release, or flag that ONE plate with model, layer interval, defect and fixes tried. The batch never stops. GO means released to Mike for inspection; only Mike authorises a print.

## 6. Where the two disagree
- ChatGPT scores the band by burst size (released area / model area) and an off-centre measure. Grok rejects magnitudes and uses the three yes/no events above, because the magnitude family did not sort the 19 plates. This rule uses Grok's events; ChatGPT's burst number is logged for information only.
- ChatGPT words GO as "no improving swap available"; Grok says a clean hard gate is a GO and the mover simply runs until no legal improving move is left. Same outcome; Grok's wording is used.

## 7. Unsolved, kept out of the gate
Lattice holes as extra peel edges; whether the last loop sits under model body or bare lattice; tall-on-small-base (the layer mask cannot see height); any force in newtons or kPa.