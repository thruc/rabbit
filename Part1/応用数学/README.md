# ラビットチャレンジ　応用数学レポート
- 線形代数
- 確率・統計
- 情報理論  

[![Bbnr_jdla.png](http://ai999.careers/bnr_jdla.png)](http://study-ai.com/jdla/)

# 線形代数
## 行列とは  

数字を四角に並べたもの

```math
\begin{pmatrix}
1 & 4 & 3\\
2 & 6 & 3\\
2 & 4 & 2
\end{pmatrix}
```
行列を計算していくことによって性質や特徴を探れる

## 連立方程式を行列で表す  

```math:
\left\{
\begin{array}{ll}
x_1  + 2x_2 = 5
\\
2x_1 + 3x_2 = 8
\end{array}
\right.
\Leftrightarrow
\begin{pmatrix}
1 & 2 \\
2 & 3
\end{pmatrix}
\begin{pmatrix}
x_1 \\ x_2
\end{pmatrix}
=
\begin{pmatrix}
5 \\ 8
\end{pmatrix}
```

ベクトルは縦で書く

## 行列とベクトル計算
先程の連立方程式との関係から  

```math
\begin{pmatrix}
1 & 2 \\
2 & 3
\end{pmatrix}
\begin{pmatrix}
1 \\ 2
\end{pmatrix}
=
\begin{pmatrix}
1\times1+2\times2 \\ 2\times1+3\times2
\end{pmatrix}
=
\begin{pmatrix}
5 \\ 8
\end{pmatrix}
```

行列が作用してベクトルが変換される

## 行列と行列の掛け算　行列の積    


```math
\begin{pmatrix}
1 & 2 \\
3 & 4
\end{pmatrix}
\times
\begin{pmatrix}
4 & 3 \\
2 & 1
\end{pmatrix}
```
行列とベクトルの掛け算から
右側の行列は２列にベクトルが並んだものと考えると


```math
\begin{pmatrix}
1 & 2 \\
3 & 4
\end{pmatrix}
\times
\begin{pmatrix}
4 & 3 \\
2 & 1
\end{pmatrix}
=
\begin{pmatrix}
1 \times 4 + 2 \times 2 & 1\times 3 + 2 \times 1 \\
3 \times 4 + 4 \times 2 & 3 \times 3 + 4 \times 1
\end{pmatrix}
=
\begin{pmatrix}
9 & 5 \\
20 & 13
\end{pmatrix}
```
行列も行列によって変換されて新しい成分を持った行列ができると考えられる

## 単位行列と逆行列 
行列の中には、その形や性質に応じていくつかの名前が付けられているものがある
#### 単位行列 
もとの行列を変化させない行列 

```math
I
=
\begin{pmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1 \\
\end{pmatrix}が単位行列
\\
すなわち行列Aにたいして\\
\\
AI=IA=Aが成り立つ
```
#### 逆行列
ある行列にかけると単位行列になるもの

```math
行列Aの逆行列は\\
A^{-1} と書き\\
A^{-1}A=AA^{-1}=I\\
が成り立つ
```
逆行列によって行列で割り算のような計算ができる

### 行列式  
逆行列が存在するか判別する式 
正方行列の大きさみたいなもの 

```math
\begin{vmatrix}
a & b \\
c & d
\end{vmatrix}
= ad - bc
```
```math
\begin{vmatrix}
a_{11} & a_{12} & a_{13}\\
a_{21} & a_{22} & a_{23}\\
a_{31} & a_{32} & a_{33}
\end{vmatrix}
= a_{11}
\begin{vmatrix}
a_{22} & a_{23}\\
a_{32} & a_{33}
\end{vmatrix} -
 a_{21}
\begin{vmatrix}
a_{12} & a_{13}\\
a_{32} & a_{33}
\end{vmatrix} + 
 a_{31}
\begin{vmatrix}
a_{12} & a_{13}\\
a_{22} & a_{23}
\end{vmatrix} 
```

### 固有値・固有値ベクトル
```math  

あるベクトル\vec{x}にある行列Aをかけたら\\
λ（定数）倍の\vec{x}が出てくるときあり\\
そのλ倍されているベクトルを固有値ベクトル\\
λを固有値とよんでいる
```
```math
式で表すと
```
```math
A\vec{x} = λ\vec{x}
```
#### 固有値・固有値ベクトルの求め方


```math
先程の式をの右辺に単位ベクトルをかける
A\vec{x} = λI\vec{x}\\
Iは単位ベクトル\\
A\vec{x} - λI\vec{x}= 0\\
(A - λI)\vec{x}= 0\\
\vec{x} \neq 0は自明な解は除く\\
ここで(A - λI)は\vec{x} \neq 0　なので逆行列をもたない\\
よって(A - λI)の行列式が０となるので\\
\begin{vmatrix}
A - λI
\end{vmatrix} = 0\\
となるのでλについて解いて固有値を求める\\
固有値をそれぞれA\vec{x} = λ\vec{x}に代入し整理し固有ベクトルを求める\\
```

### 固有値分解  

固有値・固有ベクトルの組は複数存在する場合があります。仮にn個の組があったとすると

```math
A\vec{v_1} = λ_1\vec{v_1}\\
A\vec{v_2} = λ_2\vec{v_2}\\
A\vec{v_3} = λ_3\vec{v_3}\\
 \vdots\\
A\vec{v_n} = λ_n\vec{v_n}\\
```

これを一つにまとめたいという考えから
固有ベクトルを横に並べた行列を

```math
V=\begin{pmatrix}
\vec{v_1} & \vec{v_2} & \cdots\\
\end{pmatrix}
```

対角成分に固有値を並べたものを

```math
Λ=
\begin{pmatrix}
λ_1 & 0 & \cdots & 0\\
0 & λ_2 & & \vdots \\
\vdots &  & \ddots & \vdots \\
0 & \cdots & \cdots & λ_n
\end{pmatrix}
```
とおくと

```math
AV = VΛで表せる\\
よってA = VΛV^{-1}となる。
```
上の式の右辺が正方行列Aの固有値分解したものとなる

### 特異値分解  
正方行列以外でも固有値分解をしたい
特異値分解という方法がある

```math
M \vec v = σ \vec u \\
M^T \vec u = σ \vec v \\
```
このような単位ベクトルがある場合特異値分解ができる

```math
M= USV^T\\
ただしUとVは直交行列
```
# 確率・統計  
## 確率とは  
確かさの率
どれだけ確かという考えること
## 頻度確率
発生する頻度
何度も何度も測定して必ず決まる
数値がすべて把握できる
客観確率とも言われる

## ベイズ確率
信念の度合いとも考えることができる
不確かな数値から確率を決める
主観確率とも言われている

## 確率を定義する  

```math
P(A)=\frac{n(A)}{n(U)} = \frac{事象Aが起こる数}{すべての事象の数}
```

このように書くことによって分析することができる
## 同時確率 
事象Aと事象Bが同時に起こる確率

```math
P(A\cap B)=P(A)P(B|A)\\
P(B\cap A)=P(B)P(A|B)
``` 

## 条件付き確率  
ある事象Bが与えられたもとでAとなる確率

```math
\begin{align}
P(A|B) &= \frac{P(A\cap B)}{P(B)}\\
& = \frac{n(A\cap B)}{n(B)}
\end{align}
```
## 独立な事象の同時確率
お互いの発生に関係を及ぼさない事象Aと事象Bが同時発生する

```math
P(A\cap B)=P(A)P(B|A)=P(A)P(B)\\

``` 

## 和事象の確率
事象A又は事象Bが起こる事象

```math
P(A\cup B)=P(A)+P(B)-P(A\cap B)

``` 

## 統計  
記述統計と推測統計がある

## 記述統計
記述する統計
データの数が増えたデータを整理して使用
データの全体をみるということを数理的に行う
集団の性質を要約し記述する

## 推測統計
すべてのデータを集めることが難しいとき行う
集団から一部を取り出し元の集団の性質を推測する

## 確率変数
事象と結び付けられた数値
事象そのものを指す解釈する場合も多い

## 確率分布
確率変数が出てくる確率の分布
離散値であれば表に表せる

## 期待値
概ね平均値と考えても良い

事象$X$

```math
x_1,x_2,\dots,x_n
```
確率変数$f(x)$

```math
f(x_1),f(x_2),\dots.f(x_n)
```
確率$P(x)$

```math
P(x_1),P(x_2),\dots,P(x_n)
```
のとき期待値$E(f)$は

```math

\sum_{K=1}^nP(X=x_K)f(X=x_{K})

```
連続地の場合

```math
\int P(X=x)f(X=x)dx
```
となる

## 分散 
 
データの散らばり具合
平均と各値がどれだけずれているか平均したもの
絶対値を取る不便さをなくすため２乗している
分散が大きいほどデータが散らばっている

分散$Varf(x)$は

```math
Var(f) = E
 \Bigl(
 \bigl(
f_{(X=x)}-E_{(f)}
\bigr)^2
\Bigr)
=E \bigl(f_{(X=x)}\bigr)^2- \bigl(E_{(f)} \bigr)^2
```
## 共分散  

2つのデータの傾向の違い
正の値ならば似た傾向
負の値ならば逆の傾向
ゼロならば関係性が乏しい

```math
Cov(f,g) = E
 \Bigl(
 \bigl(
f_{(X=x)}-E_{(f)}
\bigr)
 \bigl(
g_{(Y=y)}-E_{(g)}
\bigr)
\Bigr)
=E(f_{(fg)})- E(f)E(g)
```
## 標準偏差
分散が2乗をとっているため単位が変わってしまう
同じ単位のデータを比較したいので平方根をとったのが標準偏差$σ$

```math
σ=\sqrt{Var(f)}=\sqrt{E \bigl(f_{(X=x)}\bigr)^2- \bigl(E_{(f)} \bigr)^2}
```

## ベルヌーイ分布
２種類のみの結果
コイントスで歪んだコインでも扱えるようにしたもの

```math
P(x|μ) = μ^x(1-μ)^{1-x}
```
## マルチヌーイ分布
２種類以上の結果
サイコロが歪んでいても扱える
数式はなくベルヌーイ分布と同じような考え方

## 二項分布
ベルヌーイ分布の多試行版

```math
P(x|λ,n) = \frac{n!}{x!(x-n)!}λ^x(1-λ)^{n-x}
```
## ガウス分布
釣鐘型の連続分布

```math
N(x;μ,σ^2) = \sqrt{\frac{1}{2πσ^2}}exp 
\bigl(
-{\frac{1}{2σ^2}}(x-μ)^2
\bigr)
```

## 推定
母集団を特徴づけるパラメータを理論に基づいて推測すること

## 推定量
パラメータを推定するために利用する数値の計算方法や計算式のこと
推定関数ともいう
推定値を出してくれるもの
推定するもの

## 推定値
実際に試行を行なった結果から計算した値
推定されたもの

## 標本平均
母集団から取り出した標本の平均値


## 不偏分散
母集団に比べ標本数が少ない時は、標本分散が母分散よりも小さくなる
標本分散が母分散に等しくなるように補正したもの

```math
s^2 = \frac{1}{n-1}\sum_{k=1}^{n}(x_i - \bar{x})^2
```

# 情報理論
どうやって情報取り扱うか
情報を数式化して扱う

## 自己情報量
情報の珍しさ

```math
I(x) = -log(P(x))
     = log(W(x))
```

## シャノンエントロピ
自己情報量の期待値
誤差関数の中身に使う

```math
\begin{align}
H(x) &= E(I(x))\\
     &= -E(log(P(x))\\
     &= -\sum (P(x)log(P(X))\\
\end{align}
```

## カムバック・ライブラー ダイバージェンス
同じ事象・確率変数における異なる確率分布P,Qの違いを表す

```math
D_{KL}(P||Q) = E_{X\sim P}
\begin{bmatrix}
logP(x)-logQ(x)
\end{bmatrix}

```
## 交差エントロピ
KLダイバージェンスの一部を取り出したもの
Qについての自己情報量をPの分布で平均している

```math
H(P,Q) = H(P) + D_{KL}(P||Q) \\
H(P,Q) = -E_{X\sim P}  log(Q(x)
```

## 参考資料
線形代数
https://thinkit.co.jp/article/16884
情報理論
https://cookie-box.hatenablog.com/entry/2017/05/07/121607
数式の書き方
https://qiita.com/PlanetMeron/items/63ac58898541cbe81ada
