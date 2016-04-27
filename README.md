# LRCN
 Long-term Recurrent Convolutional Networks (LRCN)
 [論文](http://arxiv.org/pdf/1411.4389.pdf)

## データセット準備

1. データセットはここからダウンロードしてください (http://crcv.ucf.edu/data/UCF101.php)。
2. LRCNのディレクトリーに内に解凍する。
3. 使ってるPCでファイルが開けなかったり、OpenCVでビデオをCaptureできない場合は、```./converter.sh```でファイルをいい感じに変換してください。
4. LRCNのディレクトリー内に「images」フォルダを作成し、```python movie2image.py```を実行すると各ムービーから、時間間隔として均等に10フレーム取り出した画像を各動作の各動画毎のフォルダに保存する（半日以上かかります）。

## CNNモデルの準備

1. 今回はCaffeのAlexNetのモデルをベースとしたCNNにより画像の特徴を抽出する。そのため、まずCaffeのBVLCAlexNetモデルをダウンロードしてください(https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet)。
2. ```python caffe2chainer.py```によって、今回のネットワーク（AlexNetの最後のLinear部分を一部省略したもの）にあったモデルとして改めて保存。

## LRCNの動かし方
```python lrcn.py```

## 備考
まだまともに学習させてないのと、構造的にそもそも正しいのか現在検証中。
コードも他人に見せるようにまだ書いてないので、もう少し落ち着いてから見てもらえればと思ってます。（4/26）
