# Tabelle

| accurcy    | QSVM (simulator)                                                                   | QSVM (quantum) | SVM (classical) | QNN (simulator) | QNN (quantum) | NN (classical) | Hybrid (simulator)                                                                                        | Hybrid (quantum) |
|------------|------------------------------------------------------------------------------------|----------------|-----------------|-----------------|---------------|----------------|-----------------------------------------------------------------------------------------------------------|------------------|
| AdHoc 100  | fold1, fold2, fold3, fold4, fold5, fold6, fold7, fold8, fold9, fold10<br/>laufzeit |                |                 |                 |               |                |                                                                                                           |                  |
| AdHoc 1k   |                                                                                    |                |                 |                 |               |                |                                                                                                           |                  |
| AdHoc 10k  |                                                                                    |                |                 |                 |               |                |                                                                                                           |                  |
| VLDS 100   |                                                                                    |                |                 |                 |               |                | 1.0000, 0.9600, 0.9600, 0.9600, 0.9600, 0.9200, 0.9600, 0.9600, 0.9600, 0.9600<br/> training: 77.035424s  |                  |
| VLDS 1k    |                                                                                    |                |                 |                 |               |                | 0.8160, 0.7840, 0.8240, 0.8520, 0.8080, 0.8160, 0.8240, 0.7840, 0.8360, 0.7720<br/>training: 960.902721s  |                  |
| VLDS 10k   |                                                                                    |                |                 |                 |               |                | 0.9984, 0.9948, 0.8368, 0.9892, 0.9944, 0.9956, 0.9984, 0.9988, 0.9988, 0.9980<br/>training: 6656.448425s |                  |
| Custom 100 |                                                                                    |                |                 |                 |               |                | 0.7200, 0.7200, 0.7200, 0.7200, 0.7200, 0.7200, 0.7200, 0.7200, 0.7200, 0.7200<br/>training: 77.34316s    |                  |
| Custom 1k  |                                                                                    |                |                 |                 |               |                | 0.7280, 0.7280, 0.7280, 0.7280, 0.7280, 0.7280, 0.7280, 0.7280, 0.7280, 0.7280<br/>training: 958.979004   |                  |
| Custom 10k |                                                                                    |                |                 |                 |               |                |                                                                                                           |                  |

1. Jupyter notbook für data set ausführen
2. Werte in Tabelle übernehmen
3. Git commit
4. Notebook für 2. data set ausführen
5. Werte in Tabelle...

QSVM -> Karin
QNN -> Stefan
Hybrid -> Lukas
