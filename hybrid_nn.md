# Hybrid NN Training

## VLDS 100

**Using 100 epochs, NLLLoss, plotting neg. log. loss**

![vlds_100](https://user-images.githubusercontent.com/8350985/212035722-fd110431-7761-4cd3-9a8a-bcbeacd17526.png)

```
Fold 0 accuracy 100.00% on test set, trained in 76.78086 seconds
Fold 1 accuracy 96.00% on test set, trained in 76.07456 seconds
Fold 2 accuracy 96.00% on test set, trained in 76.21540 seconds
Fold 3 accuracy 96.00% on test set, trained in 76.56493 seconds
Fold 4 accuracy 96.00% on test set, trained in 76.62916 seconds
Fold 5 accuracy 92.00% on test set, trained in 77.82518 seconds
Fold 6 accuracy 96.00% on test set, trained in 77.80424 seconds
Fold 7 accuracy 96.00% on test set, trained in 77.39037 seconds
Fold 8 accuracy 96.00% on test set, trained in 77.30363 seconds
Fold 9 accuracy 96.00% on test set, trained in 77.76591 seconds
Accuracy mean: 96.0%, std: 1.7888543819998317
```
 
## VLDS 1k

**Using 100 epochs, NLLoss**

![image](https://user-images.githubusercontent.com/8350985/212051796-3993bc85-8e32-4ee8-9d74-dfe52955c966.png)
```
Fold 0 accuracy 81.60% on test set, trained in 959.82757 seconds
Fold 1 accuracy 78.40% on test set, trained in 962.82789 seconds
Fold 2 accuracy 82.40% on test set, trained in 961.66080 seconds
Fold 3 accuracy 85.20% on test set, trained in 962.79903 seconds
Fold 4 accuracy 80.80% on test set, trained in 961.43279 seconds
Fold 5 accuracy 81.60% on test set, trained in 958.32169 seconds
Fold 6 accuracy 82.40% on test set, trained in 961.09307 seconds
Fold 7 accuracy 78.40% on test set, trained in 962.00527 seconds
Fold 8 accuracy 83.60% on test set, trained in 960.43157 seconds
Fold 9 accuracy 77.20% on test set, trained in 958.62753 seconds
Accuracy mean: 81.16000000000001%, std: 2.3829393613770335
```

## VLDS 10k

**Using 100 epochs, CrossEntropyLoss**

![vlds_10k](https://user-images.githubusercontent.com/8350985/212074378-66191150-5468-4036-b5d7-7c46fc20642d.png)

```
Fold 0 accuracy 99.84% on test set, trained in 6764.19394 seconds
Fold 1 accuracy 99.48% on test set, trained in 6771.66353 seconds
Fold 2 accuracy 83.68% on test set, trained in 6323.10434 seconds
Fold 3 accuracy 98.92% on test set, trained in 6392.81630 seconds
Fold 4 accuracy 99.44% on test set, trained in 6749.09463 seconds
Fold 5 accuracy 99.56% on test set, trained in 6720.02865 seconds
Fold 6 accuracy 99.84% on test set, trained in 6710.57284 seconds
Fold 7 accuracy 99.88% on test set, trained in 6676.22929 seconds
Fold 8 accuracy 99.88% on test set, trained in 6702.86400 seconds
Fold 9 accuracy 99.80% on test set, trained in 6753.91673 seconds
Accuracy mean: 98.03199999999998%, std: 4.792454068637487
```

Evaluation of 10 test items on quantum computer:

```
Accuracy: 80.00%
```

## Custom 100

![custom_100](https://user-images.githubusercontent.com/8350985/212082692-6a981168-d9b5-44c7-bfd4-266992896d93.png)

**Using 100 epochs, NLLLoss**

```
Fold 0 accuracy 72.00% on test set, trained in 76.53786 seconds
Fold 1 accuracy 72.00% on test set, trained in 77.16866 seconds
Fold 2 accuracy 72.00% on test set, trained in 77.24908 seconds
Fold 3 accuracy 72.00% on test set, trained in 77.00466 seconds
Fold 4 accuracy 72.00% on test set, trained in 76.12173 seconds
Fold 5 accuracy 72.00% on test set, trained in 77.86548 seconds
Fold 6 accuracy 72.00% on test set, trained in 77.63027 seconds
Fold 7 accuracy 72.00% on test set, trained in 78.20151 seconds
Fold 8 accuracy 72.00% on test set, trained in 78.11769 seconds
Fold 9 accuracy 72.00% on test set, trained in 77.53465 seconds
Accuracy mean: 72.0%, std: 0.0
```

## Custom 1k

![image](https://user-images.githubusercontent.com/8350985/212058558-fa59d12b-1a44-4cef-abf7-c6214cc6e20a.png)

```
Fold 0 accuracy 72.80% on test set, trained in 958.67977 seconds
Fold 1 accuracy 72.80% on test set, trained in 960.01930 seconds
Fold 2 accuracy 72.80% on test set, trained in 962.09803 seconds
Fold 3 accuracy 72.80% on test set, trained in 959.60226 seconds
Fold 4 accuracy 72.80% on test set, trained in 954.58436 seconds
Fold 5 accuracy 72.80% on test set, trained in 957.46893 seconds
Fold 6 accuracy 72.80% on test set, trained in 958.87594 seconds
Fold 7 accuracy 72.80% on test set, trained in 958.93899 seconds
Fold 8 accuracy 72.80% on test set, trained in 960.49738 seconds
Fold 9 accuracy 72.80% on test set, trained in 959.02508 seconds
Accuracy mean: 72.79999999999998%, std: 1.4210854715202004e-14
```

## Custom 10k

![image](https://user-images.githubusercontent.com/8350985/212105222-bb75a8a0-1242-46e7-b3be-3901381d522a.png)

```
Fold 0 accuracy 70.32% on test set, trained in 9624.00801 seconds
Fold 1 accuracy 70.32% on test set, trained in 9627.44166 seconds
Fold 2 accuracy 70.32% on test set, trained in 9641.67902 seconds
Fold 3 accuracy 70.32% on test set, trained in 9630.13388 seconds
Fold 4 accuracy 70.32% on test set, trained in 9613.54807 seconds
Fold 5 accuracy 70.32% on test set, trained in 9632.72226 seconds
Fold 6 accuracy 70.32% on test set, trained in 9619.94806 seconds
Fold 7 accuracy 70.32% on test set, trained in 9597.62820 seconds
Fold 8 accuracy 70.32% on test set, trained in 9628.16740 seconds
Fold 9 accuracy 70.32% on test set, trained in 9608.07188 seconds
Accuracy mean: 70.32000000000002%, std: 1.4210854715202004e-14
```

## Adhoc 100

**Using 200 epochs, CrossEntropyLoss**

![adhoc_100](https://user-images.githubusercontent.com/8350985/212484560-e7d14fe5-8f07-4cb4-b46e-57e446952208.png)

```
Fold 0 accuracy 56.00% on test set, trained in 159.94229 seconds
Fold 1 accuracy 52.00% on test set, trained in 160.86594 seconds
Fold 2 accuracy 68.00% on test set, trained in 159.31214 seconds
Fold 3 accuracy 48.00% on test set, trained in 160.42371 seconds
Fold 4 accuracy 72.00% on test set, trained in 160.53258 seconds
Fold 5 accuracy 56.00% on test set, trained in 161.49501 seconds
Fold 6 accuracy 56.00% on test set, trained in 160.66374 seconds
Fold 7 accuracy 52.00% on test set, trained in 161.78155 seconds
Fold 8 accuracy 68.00% on test set, trained in 160.98317 seconds
Fold 9 accuracy 56.00% on test set, trained in 161.00550 seconds
Accuracy mean: 58.4%, std: 7.631513611335564
```

## Adhoc 1k

**Using 200 epochs, CrossEntropyLoss**

![adhoc_1k](https://user-images.githubusercontent.com/8350985/212488825-31b6575e-430b-4496-8d8b-5105112b1ea4.png)

```
Fold 0 accuracy 62.00% on test set, trained in 1461.04065 seconds
Fold 1 accuracy 56.00% on test set, trained in 1449.69642 seconds
Fold 2 accuracy 62.00% on test set, trained in 1463.15211 seconds
Fold 3 accuracy 59.20% on test set, trained in 1465.40207 seconds
Fold 4 accuracy 57.60% on test set, trained in 1467.03516 seconds
Fold 5 accuracy 50.40% on test set, trained in 1452.85427 seconds
Fold 6 accuracy 60.00% on test set, trained in 1453.56368 seconds
Fold 7 accuracy 72.00% on test set, trained in 1469.64044 seconds
Fold 8 accuracy 65.20% on test set, trained in 1463.38965 seconds
Fold 9 accuracy 62.40% on test set, trained in 1450.16790 seconds
Accuracy mean: 60.67999999999999%, std: 5.428959384633487
```

## Adhoc 10k

**Using 100 epochs, CrossEntropyLoss**

![adhoc_10k](https://user-images.githubusercontent.com/8350985/212172131-9e747a30-40da-4032-9e9a-7ea642b87b1b.png)

```
Fold 0 accuracy 62.04% on test set, trained in 6803.07014 seconds
Fold 1 accuracy 53.48% on test set, trained in 6774.40922 seconds
Fold 2 accuracy 57.64% on test set, trained in 6668.53218 seconds
Fold 3 accuracy 59.36% on test set, trained in 6843.34581 seconds
Fold 4 accuracy 50.64% on test set, trained in 6814.30222 seconds
Fold 5 accuracy 52.76% on test set, trained in 6757.89940 seconds
Fold 6 accuracy 55.16% on test set, trained in 6652.42507 seconds
Fold 7 accuracy 60.28% on test set, trained in 6844.28554 seconds
Fold 8 accuracy 52.32% on test set, trained in 6807.48312 seconds
Fold 9 accuracy 59.04% on test set, trained in 6670.76669 seconds
Accuracy mean: 56.27199999999999%, std: 3.702790299220306
```
