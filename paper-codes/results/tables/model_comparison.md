# Model comparison under a uniform 25-dimensional input

## Results

| model | target | set | n | R2 | MAE | RMSE | MAPE (%) |
|---|---|---|---|---|---|---|---|
| xgb | ZDEN | validation | 2300 | 0.9437 | 0.0107 | 0.0136 | 0.4432 |
| xgb | ZDEN | blind well (normal) | 3511 | 0.9457 | 0.0105 | 0.0132 | 0.4345 |
| xgb | DTC | validation | 2300 | 0.9409 | 0.8658 | 1.0784 | 1.0405 |
| xgb | DTC | blind well (normal) | 3511 | 0.9406 | 0.8693 | 1.0927 | 1.0391 |
| rf | ZDEN | validation | 2300 | 0.9441 | 0.0107 | 0.0135 | 0.4407 |
| rf | ZDEN | blind well (normal) | 3511 | 0.9465 | 0.0104 | 0.0131 | 0.4312 |
| rf | DTC | validation | 2300 | 0.9427 | 0.8520 | 1.0618 | 1.0241 |
| rf | DTC | blind well (normal) | 3511 | 0.9415 | 0.8635 | 1.0844 | 1.0318 |
| mlp | ZDEN | validation | 2300 | 0.9351 | 0.0114 | 0.0146 | 0.4723 |
| mlp | ZDEN | blind well (normal) | 3511 | 0.9390 | 0.0111 | 0.0140 | 0.4594 |
| mlp | DTC | validation | 2300 | 0.9402 | 0.8679 | 1.0851 | 1.0442 |
| mlp | DTC | blind well (normal) | 3511 | 0.9408 | 0.8706 | 1.0909 | 1.0418 |
| ann | ZDEN | validation | 2300 | 0.9360 | 0.0114 | 0.0145 | 0.4714 |
| ann | ZDEN | blind well (normal) | 3511 | 0.9310 | 0.0118 | 0.0149 | 0.4882 |
| ann | DTC | validation | 2300 | 0.9391 | 0.8728 | 1.0949 | 1.0490 |
| ann | DTC | blind well (normal) | 3511 | 0.9355 | 0.9062 | 1.1387 | 1.0831 |
| lstm | ZDEN | validation | 2300 | 0.9448 | 0.0106 | 0.0135 | 0.4388 |
| lstm | ZDEN | blind well (normal) | 3511 | 0.9479 | 0.0103 | 0.0129 | 0.4254 |
| lstm | DTC | validation | 2300 | 0.9434 | 0.8438 | 1.0552 | 1.0145 |
| lstm | DTC | blind well (normal) | 3511 | 0.9419 | 0.8590 | 1.0806 | 1.0261 |
| rnn | ZDEN | validation | 2300 | 0.9440 | 0.0107 | 0.0136 | 0.4414 |
| rnn | ZDEN | blind well (normal) | 3511 | 0.9479 | 0.0103 | 0.0129 | 0.4259 |
| rnn | DTC | validation | 2300 | 0.9428 | 0.8461 | 1.0610 | 1.0181 |
| rnn | DTC | blind well (normal) | 3511 | 0.9431 | 0.8540 | 1.0696 | 1.0211 |
| gru | ZDEN | validation | 2300 | 0.9446 | 0.0106 | 0.0135 | 0.4401 |
| gru | ZDEN | blind well (normal) | 3511 | 0.9481 | 0.0103 | 0.0129 | 0.4251 |
| gru | DTC | validation | 2300 | 0.9429 | 0.8477 | 1.0604 | 1.0192 |
| gru | DTC | blind well (normal) | 3511 | 0.9431 | 0.8518 | 1.0694 | 1.0176 |
| linear | ZDEN | validation | 2300 | 0.9441 | 0.0107 | 0.0136 | 0.4415 |
| linear | ZDEN | blind well (normal) | 3511 | 0.9486 | 0.0102 | 0.0128 | 0.4227 |
| linear | DTC | validation | 2300 | 0.9424 | 0.8509 | 1.0648 | 1.0231 |
| linear | DTC | blind well (normal) | 3511 | 0.9429 | 0.8573 | 1.0718 | 1.0248 |
| knn | ZDEN | validation | 2300 | 0.9412 | 0.0110 | 0.0139 | 0.4537 |
| knn | ZDEN | blind well (normal) | 3511 | 0.9413 | 0.0109 | 0.0137 | 0.4513 |
| knn | DTC | validation | 2300 | 0.9379 | 0.8810 | 1.1054 | 1.0593 |
| knn | DTC | blind well (normal) | 3511 | 0.9315 | 0.9293 | 1.1739 | 1.1103 |
| svr | ZDEN | validation | 2300 | 0.9428 | 0.0108 | 0.0137 | 0.4478 |
| svr | ZDEN | blind well (normal) | 3511 | 0.9412 | 0.0109 | 0.0137 | 0.4511 |
| svr | DTC | validation | 2300 | 0.9368 | 0.8968 | 1.1156 | 1.0780 |
| svr | DTC | blind well (normal) | 3511 | 0.9360 | 0.9077 | 1.1349 | 1.0849 |
