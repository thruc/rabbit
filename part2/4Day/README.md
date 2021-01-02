# Section1 強化学習

## 強化学習とは

長期的に報酬を最大化できるように環境の中で行動を選択できるエージェントを作ることを目標とする機械学習の一分野
→ 　行動の結果として与えられる利益（報酬）を基に、行動を決定する原理を改善していく仕組み

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/457374/3763feab-ad26-d4b5-a0b8-34b7d9ae5db9.png)

エージェント：主人公

エージェントが方針に基づいて行動しそれに見合う環境から報酬がもらえる
報酬が最大になるように方針を訓練していくイメージ

## 強化学習の応用例

マーケティングの場合
エージェント:プロフィールと購入履歴に基づいて、キャンペーンメールを送る顧客を決めるソフトウェアである。行動:顧客ごとに送信、非送信のふたつの行動を選ぶことになる。報酬:キャンペーンのコストという負の報酬とキャンペーンで生み出されると推測される売上という正の報酬を受ける

## 探索と利用のトレードオフ

利用が足りない状態 ⇔ 探索が足りない状態がトレードオフの関係
強化学習ではこれをうまく調整していく

### 探索が足りない状態

過去のデータでベストとされる行動のみを常にとり続ければ、他のさらにベストな行動を見つけることはできない

### 利用が足りない状態

未知の行動のみを常にとり続ければ、過去の経験が活かせない

## 強化学習の差分

強化学習と通常の教師あり、教師なし学習との違い
目標が違う ・教師なし、あり学習では、データに含まれるパターンを見つけ出す およびそのデータから予測することが目標
強化学習では、優れた方策を見つけることが目標

## 価値関数

価値を表す関数としては、状態価値関数と行動価値関数の 2 種類がある

### 状態価値関数

価値を決める際環境の状態の価値に注目する場合
環境の状態が良ければ価値が上がる
エージェントの行動は関係ない

```math
V^{\pi}(s)
```

### 行動価値関数

価値を決める際環境の状態と価値を組み合わせた価値に注目する場合
エージェントがある状態で行動したときの価値

```math
Q^{\pi}(s,a)
```

## 方策関数

ある環境の状態においてどのような行動をとるのか確率を与える関数

```math
\pi(s)=a
```

## 方策勾配法

方策をモデルにすることで最適化する手法

```math
\theta^{(t+1)}=\theta^{(t)}+\epsilon\nabla J(\theta)
```

```math
\nabla_{\theta}J(\theta)=\mathbb{E}_{\pi_{\theta}}[(\nabla_{\theta}log\pi_{\theta}(a|s)Q^{\pi}(s,a))]

```

$t$：時間
$\theta$：重み
$\epsilon$：学習率
$J$：誤差関数

# Section2 Alpha Go

AlphaGo Lee と AlphaGo Zero 二種類ある

## AlphaGo Lee

ValueNet と PolicyNet の CNN を利用している

### PolicyNet(方策関数)

19x19 の 2 次元データを利用
48 チャンネル持っている
19x19 の着手予想確率得られる

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/457374/7bb5fd26-10d9-7f14-bf1b-048dbc02763a.png)

### ValueNet(価値関数)

19x19 の 2 次元データを利用
49 チャンネル持っている（手番が追加）
勝率を-1 ～ 1 の範囲で得られる
勝つか負けるかの出力であるため Flatten を挟んである

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/457374/d51a8315-2c46-3970-9576-33f1da946a28.png)

### Alpha Go Lee の学習ステップ

1．教師あり学習で RollOutPolicy と PolicyNet の学習
2．強化学習で PolicyNet の学習
3．強化学習で ValueNet の学習

#### RollOutPolicy

NN ではなく線形の方策関数
探索中に高速に着手確率を出すために使用される。

### モンテカルロ木探索

コンピューター囲碁ソフトで現在もっとも有効とされている探索法

## AlphaGo Zero

### AlphaGo Lee と AlphaGo Zero の違い

1.教師あり学習を一切行わず、強化学習のみで作成 2.特徴入力からヒューリスティックな要素を排除し、石の配置のみにした
3.PolicyNet と ValueNet を１つのネットワークに統合した
4.Residual Net（後述）を導入した５、モンテカルロ木探索から RollOut シミュレーションをなくした

### PolicyValueNet

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/457374/dd892db8-1432-ce7f-9d97-904138aad91e.png)

PolicyNet と ValueNet が統合し、それぞれ方策関数と価値観数の出力が得たいため途中で枝分かれした構造の NN となる。

### Residual Block

ネットワークにショートカットを作る
ネットワークの深さを抑える
勾配消失問題が起きにくくなる
基本構造は Convolution→BatchNorm→ReLU→Convolution→BatchNorm→Add→ReLU
アンサンブル効果

### PreActivation

Residual Block の並びを BatchNor→ReLU→Convolution→BatchNorm→ReLU→Convolution→Add にして性能向上

### wideResNet

Convolution のフィルタを k 倍にした ResNet。
フィルタを増やすことで層が浅くても深い層のものと同等以上の性能を発揮

### PyramidNet

各層でフィルタ数を増やしていく ResNet

# aection3 軽量化・高速化技術

どうやってモデルを高速に学習するか
どうやって高性能ではないコンピューターでモデル動かすか

## 分散深層学習

-   深層学習は多くのデータを使用したり、パラメータ調整のために多くの時間を使用したりするため、高速な計算が求められる
-   複数の計算資源(ワーカー)を使用し、並列的にニューラルネットを構成することで、効率の良い学習を行いたい
-   データ並列化、モデル並列化、GPU による高速技術は不可欠である

## データ並列

-   親モデルを各ワーカー(コンピューターなど)に子モデルとしてコピー
-   データを分割し、各ワーカーごとに計算させる

コンピューター自体や GPU や TPU などを増やし計算を分散し学習を高速にする
データ並列化は各モデルのパラメータの合わせ方で、同期型か非同期型か決まる

### 同期型

同期型のパラメータ更新の流れ。各ワーカーが計算が終わるのを待ち、全ワーカーの勾配が出たところで勾配の平均を計算し、親モデルのパラメータを更新する。

### 非同期型

各ワーカーはお互いの計算を待たず、各子モデルごとに更新を行う。学習が終わった子モデルはパラメータサーバに Push される。新たに学習を始める時は、パラメータサーバから Pop したモデルに対して学習していく

### 同期・非同期の比較

-   処理のスピードは、お互いのワーカーの計算を待たない非同期型の方が早い
-   非同期型は最新のモデルのパラメータを利用できないので、学習が不安定になりやすい
    -> Stale Gradient Problem
-   現在は同期型の方が精度が良いことが多いので、主流となっている。

## モデル並列

-   親モデルを各ワーカーに分割し、それぞれのモデルを学習させる。全てのデータで学習が終わった後で、一つのモデルに復元
-   モデルが大きい時はモデル並列化を、データが大きい時はデータ並列化をすると良い

モデルのパラメータが多い場合ほど、効率化も向上する

## GPU による高速化

### GPGPU (General-purpose on GPU)

元々の使用目的であるグラフィック以外の用途で使用される GPU の総称

#### CPU

高性能なコアが少数
複雑で連続的な処理が得意

#### GPU

比較的低性能なコアが多数
簡単な並列処理が得意
ニューラルネットの学習は単純な行列演算が多いので、高速化が可能

### GPGPU の開発環境

#### CUDA

GPU 上で並列コンピューティングを行うためのプラットフォーム
NVIDIA 社が開発している GPU のみで使用可能
Deep Learning 用に提供されているので、使いやすい

#### OpenCL

オープンな並列コンピューティングのプラットフォーム
NVIDIA 社以外の会社(Intel, AMD, ARM など)の GPU からでも使用可能
Deep Learning 用の計算に特化しているわけではない

## 軽量化

-   量子化
-   蒸留
-   プルーニング

量子化はよく使用されている

## 量子化

ネットワークが大きくなると大量のパラメータが必要なり学習や推論に多くのメモリと演算処理が必要
→ 通常のパラメータの 64 bit 浮動小数点を 32 bit など下位の精度に落とすことでメモリと演算処理の削減を行う

数十億個のパラメータがあると重みを記憶するために多くのメモリが必要
パラメータ一つの情報の精度を落とし記憶する情報量を減らしていく
倍精度演算(64 bit)と単精度演算(32 bit)は演算性能が大きく違うため、量子化により精度を落とすことによりより多くの計算をすることができる。
16bit が無難

### メリット

計算の高速化
省メモリ化

### デメリット

精度の低下

## 蒸留

精度の高いモデルはニューロンの規模が大きなモデルとなっていてそのため、推論に多くのメモリと演算処理が必要
→ 規模の大きなモデルの知識を使い軽量なモデルの作成を行う

### モデルの簡約化

学習済みの精度の高いモデルの知識を軽量なモデルに継承させる

### メリット

蒸留によって少ない学習回数でより精度の良いモデルを作成することができる

## プルーニング

ネットワークが大きくなると大量のパラメータがすべてのニューロンが計算の精度に関係しているわけではない
→ モデルの精度に寄与が少ないニューロンを削除することでモデルの軽量化・高速化する

### ニューロン数と精度

精度にどのくらい寄与しているかの閾値を決めニューロン削除するものをきめる
閾値が高くすることによりニューロン数が減少し精度が減少する

# Section4 応用技術

実際に使用させてるモデルを紹介
※
横幅：$H$
縦幅：$W$
チャンネル：$C$
フィルタ数：$M$

## 一般的な畳み込みレイヤー

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/457374/7cc3c351-bafa-7ba0-5ab9-3c344f4c231c.png)

-   入力マップ(チャンネル数)：$H\times W\times C$
-   畳み込みカーネルサイズ：$K\times K\times C$
-   出力チャンネル数：$M$

全出力の計算量：$H\times W\times K\times K\times C\times M$
一般な畳込みレイヤーは計算量は多い

## MobileNet

画像認識モデルの軽量化版
Depthwise Convolution と Pointwise Convolution の組み合わせで軽量化を実現

### Depthwise Convolution

入力マップのチャネルごとに畳み込みを実施
出力マップをそれらと結合
通常の畳み込みカーネルは全ての層にかかっていることを考えると計算量が大幅に削減可能
各層ごとの畳み込みなので層間の関係性は全く考慮されない。通常は PW 畳み込みとセットで使うことで解決
フィルタの数（M）分計算量が削減

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/457374/6fb5c26a-cb0e-7dac-41f2-e9f66ac58e8d.png)

全出力の計算量：$H\times W\times C\times K\times K$
(一般的な全出力の計算量：$H\times W\times K\times K\times C\times M$)

### Pointwise Convolution

1 x 1 conv とも呼ばれる
入力マップのポイントごとに畳み込みを実施
出力マップ(チャネル数)はフィルタ数分だけ作成可能(任意のサイズが指定可能)
$K \times K$分の計算量が削減

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/457374/49775407-d08b-504e-b30d-d51e5a043d7a.png)
全出力の計算量：$H\times W\times C\times M$
(一般的な全出力の計算量：$H\times W\times K\times K\times C\times M$)

### まとめ

一般的な畳み込み計算を Depthwise Convolution の出力を Pointwise Convolution に分けて計算を行うことによって計算を量を削減している

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/457374/73c00204-dbdb-167e-ed5b-537279f429cb.png)

## DenseNet

画像認識のネットワーク
NN では層が深くなるにつれて、学習が難しくなるという問題があったが
ResNet などの CNN アーキテクチャでは前方の層から後方の層へアイデンティティ接続を介してパスを作ることで問題を対処した。DenseBlock と呼ばれるモジュールを用いた、DenseNet もそのようなアーキテクチャの一つである

初期の畳み込み →Dense ブロック → 変換レイヤー → 判別レイヤー
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/457374/2f4ab188-f037-cf3f-0812-a25652de1ec6.png)

### DenseBlock

出力層の前の層の竜力を足し合わせる
次第にチャンネル増える構造になっている
具体的には Batch 正規化 →Relu 関数による変換 → 畳み込み層による処理
前スライドで計算した出力に入力特徴マップを足し合わせる
入力特徴マップのチャンネル数が$l \times k$だった場合、出力は$(l+1) \times k$となる
第 l 層の出力をとすると

$$
x_1 = H_1([x_0,x_1,x_2, \dots ,x_{l-1}])
$$

一層通過するごとにｋチャンネルづつ増える場合ｋをネットワークの「growth rate」と呼ぶ

### Transition Layer

CNN では中間層でチャネルサイズを変更する（DenseBlock の入力前のサイズ戻すなど）
特徴マップのサイズを変更し、ダウンサンプリングを行うため、Transition Layer と呼ばれる層で Dence block をつなぐ

### DenseNet と ResNet の違い

DenseBlock では前方の各層からの出力全てが後方の層への入力として用いられ
RessidualBlock では前 1 層の入力のみ後方の層へ入力

## BatchNorm

レイヤー間を流れるデータの分布を、ミニバッチ単位で平均が 0・分散が 1 になるように正規化
$H \times W \times C$の sample が N 個あった場合に、N 個の**_同一チャネル_**が正規化の単位（色で単位で）
Batch Normalization はニューラルネットワークにおいて学習時間の短縮や初期値への依存低減、過学習の抑制など効果がある
###Batch Norm の問題点
Batch Size の影響をうけミニバッチのサイズを大きく取れない場合には、効果が薄くなってしまう
ハードウェアによってミニバッチ数を変更しなければならなく効果が実験しづらい
Batch Size が小さい条件下では、学習が収束しないことがあり、代わりに Layer Normalization などの正規化手法が使われることが多い

### Layer Norm

N 個の sample のうち**_一つに注目_**。$H \times W \times C$の**_全ての pixel_**が正規化の単位（画像一枚単位で）
ミニバッチの数に依存しない Batch Norm の問題点を解消

### Instance Norm

各 sample の各チャンネルごとに正規化
コントレスト正規化に寄与・画像スタイル転送やテクスチャ合成タスクなどで利用

## Wavenet

音声生成モデル → 時系列データ
生の音声波形を生成する深層学習モデル
Pixel 　 CNN を音声に応用したもの
時系列データに対して畳み込みを適用する

### Dilated comvolution

-   層が深くなるにつれて畳み込むリンクを離す
-   より広い情報を簡単に増やすことができる

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/457374/75fb106a-db94-8722-a9a7-c728527772c9.png)

## Seq2seq

系列を入力として、系列を出力するもの
入力系列が Encode（内部状態に変換）され、内部状態から Decode（系列に変換）する

-   翻訳（英語 → 日本語）
-   音声認識（波形 → テキスト）
-   チャットボット（テキスト → テキスト）

### 言語モデル

単語の並びに確率を与えるもの
数式的には同時確立を事後確率に分解して表せる

例）
You say goodbye→0.092（自然）
You say good die→0.0000032（不自然）

### RNN× 言語モデル

文章は各単語が現れる際の同時確率は事後確率で分解でき RNN で学習することによってある時点でほ次の単語を予測することできる

## 実装

https://github.com/Tomo-Horiuchi/rabbit/blob/master/part2/4Day/lecture_chap1_exercise_public.ipynb

# Trancefomer

RNN を使用していない（必要なのは Attntion だけ）
英仏の 3600 万分の学習を８ GPU で 3.5 日で完了（当時のほかのモデルよりはるかに少ない計算量）
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/457374/8baaf112-f3ed-1122-01eb-2f159f08228b.png)

-   単語ベクトルに位置情報を追加
-   複数のヘッドで Attention を計算
-   単語の位置ごとに独立処理する全結合
-   正則化し次元をまとめデコーダーへ
-   未来情報を見ないように入力され
-   入力された情報とエンコーダーの情報から予測をしていく

## Encoder-Decoder

Encoder-Decoder モデルは文章の長さに弱い

-   翻訳元の文の内容を一つのベクトルで表現
-   文章が長くなると表現が足りなくなる

## Attention

Encoder-Decoder の問題を解決する
翻訳先の単語を選択する際に、翻訳元の文中の単語の隠れ状態を利用
すべてを足すと１になるような重みを各隠れ層に分配していく
辞書オブジェクトの機能と同じような機能

### souce Target Attention

受け取った情報に対して狙うべき情報が近いかどうかで注意するものを決める

### Self Attention

自分の入力のみでどの情報に注意するかきめる

### Trancefomer Encoder

Self Attention によって文脈を考慮して各単語をエンコード

### Position Wise Feed Forwrd Networks

各 Attention 層の出力を決定
位置情報を保持したまま出力を成形
線形変換をかける層

### Scaled dot product attention

全単語に関する attention をまとめて計算する

### Multi Head attention

8 個の Scaled dot product attention の出力を合わせる
合わせたものを線形変換する
それぞれの異なる情報を収集（アンサンブル学習みたいな）

### Add

入出力の差分を学習させる
実装上は出力に入力を加算
学習・テストエラーの低減

### Norm（Layder Norm）

学習の高速化

### Position Encoding

単語の位置情報をエンコード
RNN ではないので単語列の語順を情報を追加するため

## 実装

https://github.com/Tomo-Horiuchi/rabbit/blob/master/part2/4Day/lecture_chap2_exercise_public.ipynb

## 考察

Trancefomer と Seq2seq 比べるとはるかに Trancefomer のほうが学習速度が高速で精度もよいことが分かった

# 物体認識

入力データは画像
広義の物体認識タスクは４つに分類される

| 名称         | 出力                                   | 位置     | インスタンスの区別 |
| ------------ | -------------------------------------- | -------- | ------------------ |
| 分類         | 画像に対し単一または複数のクラスラベル | 興味なし | 興味なし           |
| 物体検知     | Bounding Box                           | 興味あり | 興味なし           |
| 意味領域分割 | 各ピクセルに対し単一のクラスラベル     | 興味あり | 興味なし           |
| 個体領域分割 | 各ピクセルに対し単一のクラスラベル     | 興味あり | 興味あり           |

分類 → 物体検知 → 意味領域分割 → 個体領域分割の順で難しくなる

# 物体検知

どこに何がどんなコンフィデンスであるかを示すもの（Bounding Box）を予測する

## データセット

| 名称      | クラス | Train+Val | Box 数/画像 |
| --------- | ------ | --------- | ----------- |
| VOC12     | 20     | 11,540    | 2.4         |
| ILSVRC17  | 200    | 476,668   | 1.1         |
| MS COCO18 | 80     | 123,287   | 7.3         |
| OICOD18   | 500    | 1,743,042 | 7.0         |

Box/画像が小さいとアイコン的な映り、日常感とはかけ離れやすい
Box/画像が大きいと部分的な重なり等も見られる、日常生活のコンテキストに近い

## 評価指標

クラス分類との違いはコンフィデンスのしきい値よって BBox の数が変化する

### IoU

物体検出においてはクラスラベルだけでなく, 物体位置の予測精度も評価したい

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/457374/ef457a8d-7222-98f4-c24d-8787fa66636c.png)

Area of overlap = $TP$
Area of Union = $TP + FP + FN$

### Precision/Recal

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/457374/73efca6a-d43c-21e9-06f8-62951b9a3172.png)

コンフィデンスと IoU でそれぞれの閾値を設定する
conf.の閾値：0.5
IoU の閾値：0.5

| conf. | pred. | IoU |
| ----- | ----- | --- | ---- |
| P1    | 0.92  | 人  | 0.88 |
| P2    | 0.85  | 車  | 0.46 |
| P3    | 0.81  | 車  | 0.92 |
| P4    | 0.70  | 犬  | 0.83 |
| P5    | 0.69  | 人  | 0.76 |
| P6    | 0.54  | 車  | 0.20 |

P1:IoU > 0.5 より TP（人を検出）
P2:IoU < 0.5 より FP
P3:IoU > 0.5 より TP（車を検出）
P4:IoU > 0.5 より TP（犬を検出）
P5:IoU > 0.5 であるが既に検出済みなので FP
P6:IoU < 0.5 より FP

Precision：$\frac{3}{3+3}＝ 0.50$

Recall：$\frac{3}{0+3}＝ 1.00$

### Average Precision

conf.の閾値：$ \beta $としたとき
Precision：$R( \beta )$
Recall：$P( \beta )$
Precision-Recall curve:$P=f( R )$

Average Precision(PR 曲線の下側面積):

$$
AP =　\int_0^1 P(R)dR
$$

### FPS：Flames per Second

物体検知応用上の要請から, 検出精度に加え検出速度も問題となる

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/457374/0c8a509b-5203-4128-e7e3-8d9f22278879.png)

## セグメンテーション

### 問題点

畳み込みやプーリングによりが画像の解像度がおちてしまう
入力サイズと同じサイズで各ピクセルに対し単一のクラスラベルをつけなければならない
元のサイズに戻さなければならない →Up-sampling の壁
解決策として以下の二つがある

-   Deconvolution
-   Transposed

### Deconvolution/Transposed

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/457374/02147094-1c85-a276-164b-d5c57b5c390c.png)

上図は kernel size = 3, padding = 1, stride = 1 の Deconv.により 3×3 の特徴マップが 5×5 に Up-sampling される様子

-   通常の Conv.層と同様, カーネルサイズ・パディング・ストライドを指定
-   特徴マップの pixel 間隔を stride だけ空ける
-   特徴マップのまわりに(kernel size - 1) - padding だけ余白を作る
-   畳み込み演算を行う

逆畳み込みと呼ばれることも多いが畳み込みの逆演算ではないことに注意 → 当然, pooling で失われた情報が復元されるわけではない

### Dilated 　 Convolution

pooling を使用せず Convolution の段階で受容野を広げる工夫

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/457374/f5e8ad4e-bbd6-93d0-6a8b-83dda3854eb6.png)

3×3 の間に隙間を与える 5×5 にし受容野を広げる（rate=2）
計算量は 3×3 と同等
最終的には 15×15 に受容野を広げられる
