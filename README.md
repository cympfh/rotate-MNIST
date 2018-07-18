# Learning Rotated-degree

画像の水増し手法として、適当な範囲 (例えば U[-15,15]) でランダムに回転させるというのがある.
このとき、回転させた角度も一緒に補助的に学習対象に加えると元の画像認識精度もあがる、というのを、どっかで聞いた気がする.
論文は見つからないので気の所為だったかもしれない.

## Experiments

MNIST.
-30度から30度の範囲で一様ランダムに角度を選ぶ.
回転角度を予測させる.
ただし30で割り算して、-1から1の範囲の値だと正規化しておく.

### Results

```
   CUDA_VISIBLE_DEVICES=$(empty-gpu-device) ./main.py --learn-angle=False
Epoch 1; loss: 0.471; Test Acc: 86.82%
Epoch 2; loss: 0.320; Test Acc: 90.99%
Epoch 3; loss: 0.227; Test Acc: 94.02%
Epoch 4; loss: 0.215; Test Acc: 94.40%
Epoch 5; loss: 0.188; Test Acc: 95.02%
Epoch 6; loss: 0.154; Test Acc: 95.47%
Epoch 7; loss: 0.156; Test Acc: 95.72%
Epoch 8; loss: 0.139; Test Acc: 95.98%
Epoch 9; loss: 0.133; Test Acc: 96.25%
Epoch 10; loss: 0.112; Test Acc: 95.78%
Epoch 11; loss: 0.116; Test Acc: 96.31%
Epoch 12; loss: 0.120; Test Acc: 96.23%
Epoch 13; loss: 0.115; Test Acc: 96.34%
Epoch 14; loss: 0.109; Test Acc: 96.60%
Epoch 15; loss: 0.118; Test Acc: 96.78%
Epoch 16; loss: 0.124; Test Acc: 96.66%
Epoch 17; loss: 0.116; Test Acc: 96.70%
Epoch 18; loss: 0.119; Test Acc: 96.83%
Epoch 19; loss: 0.106; Test Acc: 96.80%
Epoch 20; loss: 0.110; Test Acc: 96.82%


   CUDA_VISIBLE_DEVICES=$(empty-gpu-device) ./main.py --learn-angle=True --angle-lr=1.0
Epoch 1; loss: 0.603; Test Acc: 87.05%
Epoch 2; loss: 0.431; Test Acc: 92.14%
Epoch 3; loss: 0.374; Test Acc: 93.65%
Epoch 4; loss: 0.311; Test Acc: 94.20%
Epoch 5; loss: 0.320; Test Acc: 95.04%
Epoch 6; loss: 0.282; Test Acc: 95.43%
Epoch 7; loss: 0.268; Test Acc: 95.72%
Epoch 8; loss: 0.268; Test Acc: 96.01%
Epoch 9; loss: 0.275; Test Acc: 96.07%
Epoch 10; loss: 0.247; Test Acc: 96.27%
Epoch 11; loss: 0.248; Test Acc: 96.31%
Epoch 12; loss: 0.222; Test Acc: 96.41%
Epoch 13; loss: 0.226; Test Acc: 96.37%
Epoch 14; loss: 0.241; Test Acc: 96.35%
Epoch 15; loss: 0.206; Test Acc: 96.60%
Epoch 16; loss: 0.218; Test Acc: 96.77%
Epoch 17; loss: 0.208; Test Acc: 96.70%
Epoch 18; loss: 0.217; Test Acc: 96.68%
Epoch 19; loss: 0.200; Test Acc: 96.59%
Epoch 20; loss: 0.211; Test Acc: 96.73%


   CUDA_VISIBLE_DEVICES=$(empty-gpu-device) ./main.py --learn-angle=True --angle-lr=0.5
Epoch 1; loss: 0.481; Test Acc: 88.39%
Epoch 2; loss: 0.377; Test Acc: 92.03%
Epoch 3; loss: 0.302; Test Acc: 93.61%
Epoch 4; loss: 0.255; Test Acc: 94.23%
Epoch 5; loss: 0.232; Test Acc: 95.00%
Epoch 6; loss: 0.235; Test Acc: 95.52%
Epoch 7; loss: 0.237; Test Acc: 95.10%
Epoch 8; loss: 0.217; Test Acc: 95.71%
Epoch 9; loss: 0.199; Test Acc: 95.82%
Epoch 10; loss: 0.178; Test Acc: 95.93%
Epoch 11; loss: 0.202; Test Acc: 95.84%
Epoch 12; loss: 0.184; Test Acc: 96.27%
Epoch 13; loss: 0.192; Test Acc: 96.20%
Epoch 14; loss: 0.188; Test Acc: 96.15%
Epoch 15; loss: 0.187; Test Acc: 96.25%
Epoch 16; loss: 0.178; Test Acc: 96.14%
Epoch 17; loss: 0.175; Test Acc: 96.52%
Epoch 18; loss: 0.163; Test Acc: 96.57%
Epoch 19; loss: 0.166; Test Acc: 96.37%
Epoch 20; loss: 0.166; Test Acc: 96.54%

   CUDA_VISIBLE_DEVICES=$(empty-gpu-device) ./main.py --learn-angle=True --angle-lr=0.1
Epoch 1; loss: 0.495; Test Acc: 85.83%
Epoch 2; loss: 0.317; Test Acc: 91.60%
Epoch 3; loss: 0.245; Test Acc: 93.83%
Epoch 4; loss: 0.202; Test Acc: 94.63%
Epoch 5; loss: 0.186; Test Acc: 94.95%
Epoch 6; loss: 0.166; Test Acc: 94.70%
Epoch 7; loss: 0.149; Test Acc: 95.36%
Epoch 8; loss: 0.149; Test Acc: 95.62%
Epoch 9; loss: 0.155; Test Acc: 95.64%
Epoch 10; loss: 0.159; Test Acc: 96.24%
Epoch 11; loss: 0.149; Test Acc: 96.25%
Epoch 12; loss: 0.124; Test Acc: 95.94%
Epoch 13; loss: 0.130; Test Acc: 96.09%
Epoch 14; loss: 0.138; Test Acc: 96.41%
Epoch 15; loss: 0.134; Test Acc: 96.56%
Epoch 16; loss: 0.124; Test Acc: 96.75%
Epoch 17; loss: 0.121; Test Acc: 96.50%
Epoch 18; loss: 0.108; Test Acc: 96.70%
Epoch 19; loss: 0.120; Test Acc: 96.33%
Epoch 20; loss: 0.109; Test Acc: 96.53%
```

全然変わらんな.
調整不足なのか、やっぱり気の所為だったかも.

