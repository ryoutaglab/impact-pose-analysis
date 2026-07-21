# impact-p-pose-analysis
格闘技インパクト解析ツール

## 概要
mma-pose-analysisの姉妹ツール
打撃インパクトに関わる3つの指標を
リアルタイムで数値化・可視化します

## 解析指標
- 腰の回転角速度（°/秒）
- 体軸の傾き角度（°）
- 左右重心バランス（%）

## 使い方

```bash
python pose_analysis_impact-p.py 動画ファイル名
```

引数を省略した場合は `sample.mp4` を解析します。Webカメラで解析する場合はカメラ番号（例: `0`）を渡してください。

解析する人数は `--persons`（または `-p`）で指定できます（省略時は2人）。

```bash
python pose_analysis_impact-p.py 動画ファイル名 --persons 1
```

引数一覧は `-h` で確認できます。

```bash
python pose_analysis_impact-p.py -h
```

## 関連リポジトリ
- mma-pose-analysis
- solo-pose-analysis

## ライセンス

[GNU Affero General Public License v3.0](LICENSE)

本ツールは [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics)（AGPL-3.0）に依存しているため、本リポジトリ全体もAGPL-3.0で公開しています。改変・再配布、およびネットワーク経由での提供（Webサービス化等）を行う場合は、AGPL-3.0の条件に従いソースコードを公開する必要があります。
