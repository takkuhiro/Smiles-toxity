# Smiles-toxity
SMILES記法で表された分子構造を用いて学習し，  
 - 毒性の有無を指定して新しい物質を生み出す*生成*言語モデル  
 - 毒性の有無を判定する*分類*モデル
 
# Features
言語Python，機械学習ライブラリPyTorch，化学構造式に関するツールrdkit使用．  
SMILES記法に関して：https://ja.wikipedia.org/wiki/SMILES記法  
今回は毒性の有無を指定しているが，これは分子構造に対してラベルをつけることができれば，それ以外でも構わない．  

# Requirement

* python 3.7.3
* rdkit
* matplotlib 3.0.3
* scikit-learn 0.20.3

 
# Usage
 
bash
git clone ~
cd src
分類：python main.py  
生成：cd rnn  
     python classify.py

毒性などのラベルがついたSMILES記法のデータを用意し，ラベルごとにファイルを作成しておく．

 
