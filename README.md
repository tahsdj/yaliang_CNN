# yanliang_CNN

## Introduction
This is project for training a cnn model to distinguish between images of chess boards, green bean cakes and manuscript papers

Here is the link of code on colab: https://colab.research.google.com/drive/16lSezFOMhdKUwjCxgR1DQejs89GWZ2Rh

## Motivation
""
「啊，好像棋盤似的。」
「我看倒有點像稿紙。」我說。
「真像一塊塊綠豆糕。」一位外號叫「大食客」的同學緊接著說。

人總會去尋求自己喜歡的事物，每個人的看法或觀點不同，並沒有什麼關係，重要的是──人與人之間，應該有彼此容忍和尊重對方的看法與觀點的雅量。

──*出自雅量*
""

機器學習也一樣，每個模型的觀點也不同，有的預測是綠豆糕有的預測像稿紙而有的分類成綠豆糕。因此當分類準確率低的時候不要氣餒，我們要培養雅量學習尊重各個模型的結果。

# Dataset
* 3 classes (chessboard, script paper, green bean cake)

| cheesboard | script paper | green bean cake |
| -------- | -------- | -------- |
|![](https://i.imgur.com/iqle9us.jpg) |![](https://i.imgur.com/4l0JhSk.jpg)|![](https://i.imgur.com/YGQXC14.jpg)|

* Training data training: 606 samples
* Test data: 202 samples
* Image size: 200x200x3


## Model

* Network in Network

## Results
* Test acc: 98.02%
* Prediction:
![](https://i.imgur.com/RG9aawH.png)
