# RS-NLOS v1.0: synthetic NLOS dataset (summary)

RS-NLOS is the dataset I built for the **non-line-of-sight (NLOS)** part of RubbleSight: detecting victims buried under rubble that no camera can see.
This page is a public summary. The data, generator code and the detection method are kept private until the dissertation is evaluated and published.

## Why synthetic data

No public dataset records a drone sensing people buried under rubble, and burying volunteers to record one would not be ethical.
Simulation also gives exact ground truth for every sample (whether a victim is present, breathing rate, burial depth, debris material, the drone's true 3-D motion), which real recordings cannot provide.

## What it contains

| Subset | Sensor | How it was generated | Size |
|---|---|---|---|
| **RS-NLOS-A** | IR-UWB radar (7.29 GHz, X4M200-class) looking down from a hovering drone | Physics-based signal model with heavy randomisation of rubble, noise and drone motion | 4,800 radar windows |
| **RS-NLOS-S** | Same radar | **NVIDIA Sionna RT** ray tracing of 300 3-D rubble scenes with ITU-R P.2040 building materials | 1,200 radar windows |
| **RS-NLOS-W** | Wi-Fi (2.4 GHz): a buried victim's phone heard by a listening drone | Sionna RT ray tracing of 800 collapsed-building scenes | 39,200 drone–phone links |

Each radar window is a 30-second recording at 20 frames/s from 1.5–4.5 m above the rubble.
The scenes vary the debris material (concrete, brick, wood, plasterboard, mixed), its thickness (2–40 cm), breathing patterns (including shallow breathing and pauses), drone hovering drift, and distractors such as vibrating debris or swaying objects.
Data is split 70/15/15 into train/validation/test, and ray-traced scenes never appear in more than one split.

## Headline findings

- **The debris material matters more than its thickness alone.** At 7.29 GHz, radar can see through roughly 25–30 cm of brick and 15–20 cm of wood, but only about 5–10 cm of concrete.
- **Breathing can be measured, not just detected.** On detected victims, the median breathing-rate error is about **0.5 breaths per minute**.
- **The drone's own hovering motion is the main obstacle for radar.** Even centimetre-scale drift hides the millimetre-scale chest movement of a breathing victim, so motion handling is a core part of the method.
- **Wi-Fi can hear most buried phones.** **91%** of drone–phone links are received above −90 dBm (concrete 76%, brick 98%, wood ~100%), falling from 95% at 5 m altitude to 83% at 30 m.
- **For Wi-Fi, timing is the bottleneck, not signal strength.** Phones only transmit probe requests every so often. The chance of catching at least one while hovering over an area is **40% in 30 s, 57% in 60 s, 74% in 2 min and 93% in 5 min**, so hover time per zone has to be planned.

## Limitations

- **Synthetic only.** These results have not yet been validated against real recordings. The next step is testing on real UWB radar recordings of people.
- Material values are uncalibrated (standard ITU-R values with a random factor for wet or dense rubble), and bodies and rubble are simplified.
- The Wi-Fi results are likely optimistic, because signals can leak out through the sides of the simulated rubble slabs.

## Release

The dataset, generator and detection method will be released after the dissertation is evaluated and published, under a licence agreed with my supervisor.
