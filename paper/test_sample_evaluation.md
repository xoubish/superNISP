# Held-out test-sample evaluation of ResNet

Run: `runs/test-evaluation-resnet-v01/evaluate_split_resnet.py` (CPU), 2026-10-18.
Checkpoint: `runs/figure5-v06/legacy-best.pt` (ResNet, epoch 49), frozen.
Protocol: identical to the paper's validation measurements: common validity mask
(JWST validity and bilinear/bicubic support), 2-sigma JWST source masks, central
105x105 crop, fixed weighted moments, GalSim adaptive moments, native NISP from
the central 21x21 pixels, 2000 paired spatial-block bootstrap draws.

The validation rerun reproduces the published table exactly (source RMSE 0.0602,
outside 0.0140; success counts 1662/1741/1786 fixed and 1767/1685/1742 adaptive).
Shape reductions here use the three-way common sample (reference, native,
ResNet), so the validation values (19.6%, 11.1%) differ slightly from the
seven-method common sample quoted in Sect. 5.2 (19.3%, 10.6%).

| Quantity | Validation (2118) | Test (2876) |
|---|---:|---:|
| Source RMSE | 0.0602 | 0.0599 |
| Outside RMSE | 0.0140 | 0.0136 |
| Fixed successes JWST/native/ResNet | 1662/1741/1786 | 2286/2399/2487 |
| Fixed reduction | 19.6 (16.9--22.2) | 18.7 (17.1--20.4) |
| Adaptive successes | 1767/1685/1742 | 2469/2341/2438 |
| Adaptive reduction | 11.1 (7.9--13.6) | 12.3 (10.3--14.3) |

Reports: `validation_report.json`, `test_report.json`; per-object arrays in
`*_measurements.npz`. Only ResNet was evaluated on the test sample.
