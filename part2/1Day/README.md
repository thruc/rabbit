#はじめに

[<img src = "http://ai999.careers/bnr_jdla.png">](http://study-ai.com/jdla/)

# 全体像

#Section1：入力層〜中間層
![DSC_0025.JPG](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/457374/9432eb88-542c-3bec-db4f-424abeb72617.jpeg)

```python
# 順伝播（単層・複数ユニット）

# 重み
W = np.array([
    [0.1, 0.2, 0.3], 
    [0.2, 0.3, 0.4], 
    [0.3, 0.4, 0.5],
    [0.4, 0.5, 0.6]
])

## 試してみよう_配列の初期化
#W = np.zeros((4,3))
#W = np.ones((4,3))
#W = np.random.rand(4,3)
#W = np.random.randint(5, size=(4,3))

print_vec("重み", W)

# バイアス
b = np.array([0.1, 0.2, 0.3])
print_vec("バイアス", b)

# 入力値
x = np.array([1.0, 5.0, 2.0, -1.0])
print_vec("入力", x)


#  総入力
u = np.dot(x, W) + b
print_vec("総入力", u)

# 中間層出力
z = functions.sigmoid(u)
print_vec("中間層出力", z)
```
###入力層


$$ 
u=w_1x_1+w_2x_2+w_3x_3+w_4x_4+b=Wx+b
$$
上記の数式の部分は 　

```python
u = np.dot(x, W) + b
print_vec("総入力", u)
```
データ　入力を受け取る場所 
どの値をどのくらい使うかを決める重みを付与させる
入力も重みも行列で表現できる
入力の全体をずらすバイアス

#### 動物種類分類ネットワークの例

![DSC_0026.jpg](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/457374/8575690c-0f9e-35fe-7766-01c829d1bb3b.jpeg)



#### 3層のニューラルネットワークのイメージとソースコード

![三層.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/457374/ac9caee7-bf8c-3281-523b-f9d79c0d12a1.png)

```python
import numpy as np
from common import functions
"""
# functions.py
import numpy as np
def relu(x):
    return np.maximum(0, x)

"""

# ウェイトとバイアスを設定
# ネートワークを作成
def init_network():
    print("##### ネットワークの初期化 #####")
    network = {}
    
    
    network['W1'] = np.array([
        [0.1, 0.3, 0.5],
        [0.2, 0.4, 0.6]
    ])
    network['W2'] = np.array([
        [0.1, 0.4],
        [0.2, 0.5],
        [0.3, 0.6]
    ])
    network['W3'] = np.array([
        [0.1, 0.3],
        [0.2, 0.4]
    ])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['b2'] = np.array([0.1, 0.2])
    network['b3'] = np.array([1, 2])

    print_vec("重み1", network['W1'] )
    print_vec("重み2", network['W2'] )
    print_vec("重み3", network['W3'] )
    print_vec("バイアス1", network['b1'] )
    print_vec("バイアス2", network['b2'] )
    print_vec("バイアス3", network['b3'] )

    return network

# プロセスを作成
# x：入力値
def forward(network, x):
    
    print("##### 順伝播開始 #####")

    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    # 1層の総入力
    u1 = np.dot(x, W1) + b1
    
    # 1層の総出力
    z1 = functions.relu(u1)
    
    # 2層の総入力
    u2 = np.dot(z1, W2) + b2
    
    # 2層の総出力
    z2 = functions.relu(u2)

    # 出力層の総入力
    u3 = np.dot(z2, W3) + b3
    
    # 出力層の総出力
    y = u3
    
    print_vec("総入力1", u1)
    print_vec("中間層出力1", z1)
    print_vec("総入力2", u2)
    print_vec("出力1", z1)
    print("出力合計: " + str(np.sum(z1)))

    return y, z1, z2

# 入力値
x = np.array([1., 2.])
print_vec("入力", x)

# ネットワークの初期化
network =  init_network()

y, z1, z2 = forward(network, x)
```


# Section2：活性化関数
活性化関数の効果で七分の出力は弱く、一部は強く伝搬させる。
特徴をよりよく伝搬させる

## ステップ関数
### ソースコード 

```python
import numpy as np
def step(x):
    return np.where( x > 0, 1, 0) 
``` 

### 数式
$$ f(x) = \left\\{ \begin{array} \\\
1 & (x \geq 0) \\\
0 & (x \lt 0) \\\
\end{array} \right. $$ 


### 特徴
- 閾値を超えたら発火する関数であり、出力は常に1か0である。
- 0〜1の間の値を表現できず、線形分離可能なものしか学習できない。

## シグモイド関数

### ソースコード 

```python
import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))
```
 
### 数式 
$$ f(u) = \frac{1}{1+e^{-u}} $$

### 特徴
- 0〜1の間を緩やかに変化する関数で、ステップ関数ではON/OFFしかない状態に対し、信号の強弱を伝えられるようになり、ニューラルネットワーク普及のきっかけとなった。
- 緩やかに変化する関数では微分が可能
- 大きな値では出力の変化が微小なため、勾配消失問題を引き起こすことがある。0になることはないので、計算リソースが常に食われてしまう。



## ReLU関数 
### ソースコード 

```python
import numpy as np
def relu(x):
    return np.maximum(0,x)

``` 
### 数式 

$$ f(x) = \left\\{ \begin{array} \\\
x & (x > 0) \\\
0 & (x \leq 0) \\\
\end{array} \right. $$  

### 特徴
- 今最も使用されている活性化関数
- 勾配消失問題の回避と０小さい時常に出力が０なのでスパース化に貢献することで良い成果をもたらしている




# Section3：出力層 
- 各クラスの確率を出す 

## 誤差関数 

- 出力の値と正解の値を比べどのくらい合っているかを表現 

### 数式（例：残差平方和）
$$ E_n(w) = \frac{1}{2} \sum_{j=1}^J(y_j-d_j)^2 = \frac{1}{2} ||(y-d)||^2 $$
全て足し合わせると０になってしまうので２乗して足し合わせている
$\frac{1}{2}$ は微分をしたとき係数が１となり計算しやすい

## 出力層の活性化関数 

### 中間層の活性化関数との違い 

#### 値の強弱 
- 中間層：閾値の前後で信号の強弱を調整
- 出力層；信号の大きさはそのまま 

#### 確率出力 
- 分類問題の場合出力層は０～１の範囲に限定し総和を１とする必要がある 

|             | 回帰     | 二値分類                    | 多クラス分類                                    |
|:-----------:|:--------:|:-------------------------:|:---------------------------------------------:|
| 活性化関数    | 恒等写像  | シグモイド関数                | ソフトマックス関数                                 |
| 活性化関数(式)  | $f(u)=u$ | $f(u)=\frac{1}{1+e^{-u}}$ | $f(i,u)=\frac{e^{u_i}}{\sum_{k=1}^K e^{u_k}}$ |
| 誤差関数       | 二乗誤差  | 交差エントロピー               | 交差エントロピー                                  |

#### データサンプルあたりの誤差
- 二乗誤差
$$
E_n(w) = \frac{1}{2} \sum_{i=1}^I(y_n-d_n)^2 
$$
- 交差エントロピー 
$$
E_n(w) = - \sum_{i=1}^Id_ilogy_i
$$

#### 学習サイクルあたりの誤差 

$$
E(w)= \sum_n^NE_n
$$

### ソフトマックス関数
#### コード

```python
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)  # オーバーフロー対策
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
```

#### 数式 
$$
f(i,u)=\frac{e^{u_i}}{\sum_{k=1}^K e^{u_k}}
$$

### 二乗誤差 

#### コード 

```python
def mean_squared_error(d, y):
    return np.mean(np.square(d - y)) / 2
```

### 交差エントロピー 

#### コード 

```python
def cross_entropy_error(d, y):
    if y.ndim == 1:
        d = d.reshape(1, d.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if d.size == y.size:
        d = d.argmax(axis=1)
             
    batch_size = y.shape[0]
    # + 1e-7は０にならないようにしている
    return -np.sum(np.log(y[np.arange(batch_size), d] + 1e-7)) / batch_size
```


# Section4：勾配降下法

```python
    y = 3 * x[0] + 2 * x[1]
    return y

# 初期設定
def init_network():
    # print("##### ネットワークの初期化 #####")
    network = {}
    nodesNum = 10
    network['W1'] = np.random.randn(2, nodesNum)
    network['W2'] = np.random.randn(nodesNum)
    network['b1'] = np.random.randn(nodesNum)
    network['b2'] = np.random.randn()

    # print_vec("重み1", network['W1'])
    # print_vec("重み2", network['W2'])
    # print_vec("バイアス1", network['b1'])
    # print_vec("バイアス2", network['b2'])

    return network

# 順伝播
def forward(network, x):
    # print("##### 順伝播開始 #####")
    
    W1, W2 = network['W1'], network['W2']
    b1, b2 = network['b1'], network['b2']
    u1 = np.dot(x, W1) + b1
    z1 = functions.relu(u1)
    
    ## 試してみよう
    #z1 = functions.sigmoid(u1)
    
    u2 = np.dot(z1, W2) + b2
    y = u2
    
    return z1, y

# 誤差逆伝播
def backward(x, d, z1, y):
    # print("\n##### 誤差逆伝播開始 #####")    

    grad = {}
    
    W1, W2 = network['W1'], network['W2']
    b1, b2 = network['b1'], network['b2']

    # 出力層でのデルタ
    delta2 = functions.d_mean_squared_error(d, y)
    # b2の勾配
    grad['b2'] = np.sum(delta2, axis=0)
    # W2の勾配
    grad['W2'] = np.dot(z1.T, delta2)
    # 中間層でのデルタ
    #delta1 = np.dot(delta2, W2.T) * functions.d_relu(z1)

    ## 試してみよう
    delta1 = np.dot(delta2, W2.T) * functions.d_sigmoid(z1)

    delta1 = delta1[np.newaxis, :]
    # b1の勾配
    grad['b1'] = np.sum(delta1, axis=0)
    x = x[np.newaxis, :]
    # W1の勾配
    grad['W1'] = np.dot(x.T, delta1)
   

    return grad

# サンプルデータを作成
data_sets_size = 100000
data_sets = [0 for i in range(data_sets_size)]

for i in range(data_sets_size):
    data_sets[i] = {}
    # ランダムな値を設定
    data_sets[i]['x'] = np.random.rand(2)
    
    ## 試してみよう_入力値の設定
    # data_sets[i]['x'] = np.random.rand(2) * 10 -5 # -5〜5のランダム数値
    
    # 目標出力を設定
    data_sets[i]['d'] = f(data_sets[i]['x'])
    
losses = []
# 学習率
learning_rate = 0.07

# 抽出数
epoch = 1000

# パラメータの初期化
network = init_network()
# データのランダム抽出
random_datasets = np.random.choice(data_sets, epoch)

# 勾配降下の繰り返し
for dataset in random_datasets:
    x, d = dataset['x'], dataset['d']
    z1, y = forward(network, x)
    grad = backward(x, d, z1, y)
    # パラメータに勾配適用
    for key in ('W1', 'W2', 'b1', 'b2'):
        network[key]  -= learning_rate * grad[key]

    # 誤差
    loss = functions.mean_squared_error(d, y)
    losses.append(loss)

print("##### 結果表示 #####")    
lists = range(epoch)


plt.plot(lists, losses, '.')
# グラフの表示
plt.show()
```

- 深層学習の目的は、学習を通して誤差を最小にするネットワークを作成すること
- 誤差 $E(w)$ を最小化するパラメータ $w$ を発見すること

### 勾配降下法

全サンプルの平均誤差
#### 数式 

$$
W^{(t+1)}=W^t-\epsilon\Delta E
$$

$$
\text{誤差勾配} \quad \nabla E = \frac{\partial E}{\partial w} = \Bigl[\frac{\partial E}{\partial w_1}…\frac{\partial E}{\partial w_M}\Bigl]
$$ 

#### ソースコード（勾配降下法）

```python

for key in ('W1', 'W2', 'b1', 'b2'):
        network[key]  -= learning_rate * grad[key]
```


### 確率的勾配降下法(SGD)
ランダムに抽出したサンプルの誤差　
データが冗長な場合の計算コストの軽減。望まない局所極小解に収束するリスクの軽減。オンライン学習ができる。
#### 数式 
$w^{(t+1)}=w^{(t)}-\epsilon\nabla{E_n}$

### ミニバッチ勾配降下法
現在一般的な方法
ランダムに分割したデータの集合（ミニバッチ）$D_t$に属するサンプルの平均誤差 
確率的勾配法のメリットを損なわず、計算機の計算資源を有効利用できる

#### 数式 
$w^{(t+1)}=w^{(t)}-\epsilon\nabla{E_t} $
$E_t=\frac{1}{N_t}\sum_{n \in D_t}{E_n}$
$N_t= D_t $

#### オンライン学習
学習データが入ってくるたびに都度パラメータを更新し、学習を進めていく方法 
#### バッチ学習
一度にすべての学習データを使ってパラメータ更新を行う

## 誤差勾配の計算 

$$
W^{(t+1)}=W^t-\epsilon\Delta E
$$

#### 数値微分 
プログラムで微小な数値を生成し疑似的に微分を計算する一般的な手法 

$ \frac{\partial E}{\partial w_m} = \frac{E(w_m+h)-E(w_m-h)}{2h}$ 

デメリット 
- 各パラメータ$w_m$それぞれについて計算するため計算の負荷が大きい
- 無駄が多い

よって

__誤差逆伝播法__を利用する




#Section5：誤差逆伝播法
算出された誤差を出力層側から順に微分し、前の層前の層へと伝播し、
最小の計算で各パラメータでの微分地を解析的に計算する 

```python
def backward(x, d, z1, y):
    # print("\n##### 誤差逆伝播開始 #####")    

    grad = {}
    
    W1, W2 = network['W1'], network['W2']
    b1, b2 = network['b1'], network['b2']

    # 出力層でのデルタ
    delta2 = functions.d_mean_squared_error(d, y)
    # b2の勾配
    grad['b2'] = np.sum(delta2, axis=0)
    # W2の勾配
    grad['W2'] = np.dot(z1.T, delta2)
    # 中間層でのデルタ
    #delta1 = np.dot(delta2, W2.T) * 
```