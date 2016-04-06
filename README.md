## 説明
[chainer-DCGAN](https://github.com/mattya/chainer-DCGAN)をpython3化させたもの。chainer 1.6以降が必要。


## 使い方
imagesフォルダ以下にサブフォルダを作りそこに画像を放り込む。モデル名と画像フォルダ名が同一で良い場合は、--image-dirは省略可。その他、エポック数、バッチサイズ、GPU使用等はオプションで指定。
```
python train.py --model model_name --image_dir image_dir --gpu 0
```

jsonに現在のエポック等を保存してあるので、途中から再開したい際は以下のように単純にモデル名だけを指定すればよい。
```
python train.py --model model_name
```
