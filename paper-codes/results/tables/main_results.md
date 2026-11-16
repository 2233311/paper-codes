# Main results: CBD-PIF-XGBoost versus plain XGBoost

## Non-enlarged accuracy

| model | target | set | n | R2 | MAE | RMSE | MAPE (%) |
|---|---|---|---|---|---|---|---|
| CBD-PIF-XGBoost | ZDEN | validation | 2300 | 0.9440 | 0.0107 | 0.0136 | 0.4414 |
| CBD-PIF-XGBoost | ZDEN | blind well (normal) | 3511 | 0.9458 | 0.0105 | 0.0132 | 0.4346 |
| CBD-PIF-XGBoost | DTC | validation | 2300 | 0.9412 | 0.8635 | 1.0761 | 1.0380 |
| CBD-PIF-XGBoost | DTC | blind well (normal) | 3511 | 0.9401 | 0.8735 | 1.0973 | 1.0439 |
| XGBoost (25D, no gate) | ZDEN | validation | 2300 | 0.9437 | 0.0107 | 0.0136 | 0.4432 |
| XGBoost (25D, no gate) | ZDEN | blind well (normal) | 3511 | 0.9457 | 0.0105 | 0.0132 | 0.4345 |
| XGBoost (25D, no gate) | DTC | validation | 2300 | 0.9409 | 0.8658 | 1.0784 | 1.0405 |
| XGBoost (25D, no gate) | DTC | blind well (normal) | 3511 | 0.9406 | 0.8693 | 1.0927 | 1.0391 |

## Enlarged direction and deviation

| model | target | set | n | up (%) | down (%) | reference gate coverage | mean predicted | mean raw | reference mean | mean |dev| |
|---|---|---|---|---|---|---|---|---|---|---|
| CBD-PIF-XGBoost | ZDEN | training wells (enlarged) | 494 | 99.3927 | 0.6073 | 0.9393 | 2.4233 | 2.3266 | 2.4236 | 0.0274 |
| CBD-PIF-XGBoost | ZDEN | blind well (enlarged) | 486 | 97.3251 | 2.6749 | 0.9609 | 2.4331 | 2.3554 | 2.4141 | 0.0300 |
| CBD-PIF-XGBoost | DTC | training wells (enlarged) | 494 | 14.3725 | 85.6275 | 0.9393 | 82.9603 | 84.4472 | 83.0681 | 1.7476 |
| CBD-PIF-XGBoost | DTC | blind well (enlarged) | 486 | 27.7778 | 72.2222 | 0.9609 | 82.4673 | 83.5147 | 83.7874 | 2.1013 |
| XGBoost (25D, no gate) | ZDEN | training wells (enlarged) | 494 | 95.9514 | 4.0486 | 0.9393 | 2.3696 | 2.3266 | 2.4236 | 0.0615 |
| XGBoost (25D, no gate) | ZDEN | blind well (enlarged) | 486 | 95.6790 | 4.3210 | 0.9609 | 2.4011 | 2.3554 | 2.4141 | 0.0347 |
| XGBoost (25D, no gate) | DTC | training wells (enlarged) | 494 | 46.9636 | 53.0364 | 0.9393 | 84.3228 | 84.4472 | 83.0681 | 2.5822 |
| XGBoost (25D, no gate) | DTC | blind well (enlarged) | 486 | 46.7078 | 53.2922 | 0.9609 | 83.3164 | 83.5147 | 83.7874 | 2.2885 |

## Blind-well CBD bins, ZDEN

| CBD bin (in) | n | raw ZDEN | CBD-PIF-XGBoost ZDEN | XGBoost (25D, no gate) ZDEN |
|---|---|---|---|---|
| [0.5, 1) | 147 | 2.3849 | 2.4205 | 2.4084 |
| [1, 2) | 154 | 2.3788 | 2.4430 | 2.4085 |
| [2, 3) | 104 | 2.3360 | 2.4467 | 2.3945 |
| [3, 10) | 62 | 2.2427 | 2.4192 | 2.3703 |

## Blind-well CBD bins, DTC

| CBD bin (in) | n | raw DTC | CBD-PIF-XGBoost DTC | XGBoost (25D, no gate) DTC |
|---|---|---|---|---|
| [0.5, 1) | 147 | 83.7075 | 83.1273 | 83.9122 |
| [1, 2) | 154 | 82.1774 | 81.7750 | 82.3756 |
| [2, 3) | 104 | 82.9924 | 81.6909 | 82.5580 |
| [3, 10) | 62 | 87.5995 | 83.7232 | 85.5772 |
