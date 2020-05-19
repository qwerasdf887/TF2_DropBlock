# TF2_DropBlock
Implementation DropBlock in Tensorflow 2

實現DropBlock的Tensorflow 2-Keras版本

將此包成一個custom layer，直接import即可使用。

## 環境

1. Tensorflow 2.1~2.2
2. Python 3.5~3.7

## 論文重點

計算出 γ drop rate。(要與原始drop rate丟棄相同數量的點)。

number of tensor * drop rate = γ * (block_size ** 2) * (feat_size ** 2)

由此求出γ，再利用γ生成mask tensor


## run
直接執行此檔則可分別看到training phase 與 testing phase的不同輸出結果。
```bashrc
python dropblock.py
```