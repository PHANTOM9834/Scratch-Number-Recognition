# Scratch-
ビジュアルプログラミング言語であるScratchで数字を認識して出力するニューラルネットワークのプロジェクトとPythonで書かれた学習用のスクリプトおよびScratchの推論用のスプライトを作成してくれるコードが入っています。
- Github, Python, Scratchを使い慣れている人向けに説明しています。慣れていない人は以下の記事を見てください。詳細の説明をしています。
- [このリポジトリの内容の使用方法]()
- [PythonをWindowsに入れる方法](https://zenn.dev/phantm9834/articles/2b322264323436)
# 文字認識.sb3の使用方法
1. Githubの`文字認識.sb3`をダウンロードする
2. [Scratch](https://scratch.mit.edu/projects/editor/?tutorial=getStarted)を開く
3. 先程ダウンロードした`文字認識.sb3`を読み込
4. 実行する(認証を押したら結果が上部中央に出ます。1秒後に文字が消されて再度文字認識を使えま)
# scratch_learning_local.pyの使い方
1. `scratch_learning_local.py`をダウンロードする
2. 必要なPythonライブラリを入れる
以下のコマンドで可能
```bash
pip install torch torchvision numpy cattorch certifi
```
3. 実行する
コマンドプロンプトで行う場合
```bash
#C:\Users\user\Downloads\scratch_learning.py ここの部分は先程ダウンロードしたファイルのパスになります。個人によって変わります
python C:\Users\user\Downloads\scratch_learning.py
```
4. 生成されたスプライトを見つける
今回はコマンドプロンプトで実行したので以下の場所にあります。
コマンドプロンプトの最初の文字、作者の場合は`C:\Users\user>`の`>`を除いた`C:\Users\user` + ` build\mnist_mlp.sprite3`となります。
フルパスは`C:\Users\user\build\mnist_mlp.sprite3`です。
ダウンロードフォルダ等に移すことをおすすめします。
# scratch_learning_googlecolab.ipynbの使い方
1. `scratch_learning_googlecolab.ipynb`をダウンロードする
2. Google Colabを開く
[Google Colab](https://colab.research.google.com/)
3. ログインして、ダウンロードしたファイルを開く
4. 上部のメニューのすべてのセルを実行を押す
5. スプライトのダウンロード
左側の「ファイル」のアイコンを押して`build`というフォルダの中にスプライトが入っています。  
ここで丸が三つ垂直に連なっているマークを押してダウンロードを選択したらダウンロードがされます。  
