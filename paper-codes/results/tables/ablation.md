# Component ablation (gating x physics-inspired features)

## Results

| model | target | blind R2 | blind MAE | Well4 enlarged up (%) | Well4 enlarged down (%) | Well4 reference gate coverage | training wells down (%) |
|---|---|---|---|---|---|---|---|
| XGB, gate=off, PIF=off | ZDEN | 0.9457 | 0.0105 | 95.6790 | 4.3210 | 0.9609 | 4.0486 |
| XGB, gate=off, PIF=off | DTC | 0.9406 | 0.8693 | 46.7078 | 53.2922 | 0.9609 | 53.0364 |
| XGB, gate=off, PIF=on | ZDEN | 0.9458 | 0.0105 | 96.7078 | 3.2922 | 0.9609 | 3.2389 |
| XGB, gate=off, PIF=on | DTC | 0.9401 | 0.8735 | 47.5309 | 52.4691 | 0.9609 | 51.0121 |
| XGB, gate=on, PIF=off | ZDEN | 0.9457 | 0.0105 | 97.3251 | 2.6749 | 0.9609 | 0.8097 |
| XGB, gate=on, PIF=off | DTC | 0.9406 | 0.8693 | 29.0123 | 70.9877 | 0.9609 | 83.1984 |
| XGB, gate=on, PIF=on | ZDEN | 0.9458 | 0.0105 | 97.3251 | 2.6749 | 0.9609 | 0.6073 |
| XGB, gate=on, PIF=on | DTC | 0.9401 | 0.8735 | 27.7778 | 72.2222 | 0.9609 | 85.6275 |
| RF, gate=off, PIF=off | ZDEN | 0.9465 | 0.0104 | 98.9712 | 1.0288 | 0.9609 | 0.2024 |
| RF, gate=off, PIF=off | DTC | 0.9415 | 0.8635 | 45.8848 | 54.1152 | 0.9609 | 58.5020 |
| RF, gate=off, PIF=on | ZDEN | 0.9465 | 0.0104 | 98.9712 | 1.0288 | 0.9609 | 0.2024 |
| RF, gate=off, PIF=on | DTC | 0.9415 | 0.8640 | 45.4733 | 54.5267 | 0.9609 | 57.8947 |
| RF, gate=on, PIF=off | ZDEN | 0.9465 | 0.0104 | 97.9424 | 2.0576 | 0.9609 | 0.0000 |
| RF, gate=on, PIF=off | DTC | 0.9415 | 0.8635 | 26.3374 | 73.6626 | 0.9609 | 86.4372 |
| RF, gate=on, PIF=on | ZDEN | 0.9465 | 0.0104 | 97.9424 | 2.0576 | 0.9609 | 0.0000 |
| RF, gate=on, PIF=on | DTC | 0.9415 | 0.8640 | 26.9547 | 73.0453 | 0.9609 | 86.0324 |
