# flowimds 利用ガイド

## 概要

`flowimds` は、小さな再利用可能なステップを組み合わせて画像処理パイプラインを構築できるライブラリです。パイプラインは入力探索からステップ適用、結果の保存までをまとめて扱います。実行後は `PipelineResult` オブジェクトが返り、次の情報を提供します。

- `processed_count`: 正常に処理できた画像の枚数
- `failed_count`: 失敗した画像の枚数
- `failed_files`: 処理に失敗したファイルパスのリスト
- `output_mappings`: 入力ファイルと出力ファイルの対応表
- `duration_seconds`: 処理にかかった総時間（秒）
- `settings`: 実行時に使用された設定のスナップショット

## パイプラインを実行する

### `Pipeline.run` でディレクトリ全体を処理する

```python
import flowimds as fi

pipeline = fi.Pipeline(
    steps=[
        fi.ResizeStep((512, 512)),
        fi.GrayscaleStep(),
        fi.DenoiseStep(mode="median", kernel_size=5),
    ],
    input_path="/path/to/input",  # str でも pathlib.Path でも可
    output_path="/path/to/output",
    recursive=True,
    preserve_structure=True,
)

result = pipeline.run()
print(f"Processed {result.processed_count} images in {result.duration_seconds:.2f}s")
```

この形式は、ライブラリにディレクトリ走査を任せたいときに便利です。サポートされる拡張子（`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif`）を自動で収集します。`recursive=True` でサブディレクトリをたどり、`preserve_structure=True` で入力の階層構造を出力側に再現できます。

### ファイルリストに対して実行する

```python
import flowimds as fi

paths = [
    "samples/input/receipt.png",
    "samples/input/avatar.jpg",
]

pipeline = fi.Pipeline(
    steps=[fi.ResizeStep((256, 256)), fi.BinarizeStep(mode="otsu")],
    input_path="samples/input",
    output_path="samples/output",
)

result = pipeline.run_on_paths(paths)
for mapping in result.output_mappings:
    print(f"{mapping.input_path} -> {mapping.output_path}")
```

`run_on_paths` は対象ファイルを事前に把握している場合や、複数のディレクトリにまたがる入力をまとめて処理したい場合に有効です。

### `run_on_arrays` でメモリ内だけで完結させる

```python
import flowimds as fi
import numpy as np

images = [np.zeros((128, 128, 3), dtype=np.uint8) for _ in range(4)]

def brighten(image: np.ndarray) -> np.ndarray:
    # apply(image) -> image を実装するオブジェクトなら独自ステップとして利用可能
    return np.clip(image + 40, 0, 255).astype(image.dtype)

pipeline = fi.Pipeline(
    steps=[fi.GrayscaleStep(), brighten],
)

transformed = pipeline.run_on_arrays(images)
print(f"Got {len(transformed)} transformed images")
```

`run_on_arrays` はファイルシステムを使わずに NumPy 配列だけを処理します。入力イテラブルの各要素が NumPy 配列かどうかを検証し、同じ順序で変換結果のリストを返します。

`run_on_arrays` のみを利用する場合、入出力パスは省略可能です。同じインスタンスで後から `run()` や `run_on_paths()` を呼ぶときは、既存ディレクトリを指す `input_path` / `output_path` を指定しておきましょう。そうしないと、入力探索や出力保存のタイミングでエラーになります。

### `PipelineResult` を活用する

```python
def summarise(result: fi.PipelineResult) -> None:
    print(f"✅ processed: {result.processed_count}")
    print(f"⚠️ failed: {result.failed_count}")
    if result.failed_files:
        print("Failed files:")
        for path in result.failed_files:
            print(f"  - {path}")
    for mapping in result.output_mappings:
        print(f"Saved {mapping.input_path.name} to {mapping.output_path}")

result = pipeline.run()
summarise(result)
```

`settings` には実行時設定（入力・出力ディレクトリ、再帰フラグなど）が含まれるので、ログや監査用途に便利です。

## パイプラインの構成

パスは `str` と `pathlib.Path` のどちらでも指定できます。主な設定項目を下表にまとめます。

| 設定項目 | 型 | 説明 |
| --- | --- | --- |
| `steps` | `PipelineStep` の反復可能オブジェクト | 各画像に順番に適用される変換のリスト。`apply(image)` を持つオブジェクトなら自作ステップも使えます。 |
| `input_path` | `str` または `Path`（任意） | `run` 使用時に画像を探索するディレクトリ。`run_on_arrays` だけ使う場合は省略可能です。 |
| `output_path` | `str` または `Path`（任意） | 変換後のファイルを書き出すディレクトリ。`run` / `run_on_paths` を使うときは必須ですが、`run_on_arrays` のみなら省略できます。 |
| `recursive` | `bool` | 画像収集時にサブディレクトリも走査するかどうか。 |
| `preserve_structure` | `bool` | `True` の場合、入力の階層構造を `output_path` 配下に再現します。`False` ならすべて直下に保存されます。 |

`steps` の順序はそのまま処理順序になります。前のステップの出力が次のステップの入力として渡される点に注意してください。

## 標準ステップリファレンス

### `ResizeStep`

- 目的: すべての画像を固定サイズ `(width, height)` に変換
- コンストラクタ: `ResizeStep(size: tuple[int, int])`
- メモ: 正の整数ペアを検証してから OpenCV の `cv2.resize`（既定はバイリニア）を使用します。

### `GrayscaleStep`

- 目的: カラー画像を単一チャンネルのグレースケールに変換
- コンストラクタ: `GrayscaleStep()`
- メモ: 2D / 3D 配列を受け付け、入力の dtype を保持したまま OpenCV で変換します。

### `BinarizeStep`

- 目的: 画像を二値化（Otsu または固定しきい値）
- コンストラクタ: `BinarizeStep(mode="otsu", threshold=None, max_value=255)`
- メモ: `mode="otsu"` は最適なしきい値を自動計算し、`mode="fixed"` は `threshold`（0〜`max_value`）の指定が必須です。出力は入力の dtype を維持します。

### `DenoiseStep`

- 目的: メディアンまたはバイラテラルフィルタでノイズを低減
- コンストラクタ: `DenoiseStep(mode="median", kernel_size=3, diameter=9, sigma_color=75.0, sigma_space=75.0)`
- メモ: `mode="median"` では `kernel_size` が奇数かつ 3 以上である必要があります。`mode="bilateral"` では `diameter`, `sigma_color`, `sigma_space` が正の値である必要があります。

### `RotateStep`

- 目的: 指定角度で画像を反時計回りに回転
- コンストラクタ: `RotateStep(angle, expand=True, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101)`
- メモ: 90 度の倍数は `numpy.rot90` を利用して高速に処理します。`expand=False` の場合は元のキャンバスサイズを維持するため、切り抜きが発生する可能性があります。ボーダーモードで未定義領域の埋め方を制御できます。

### `FlipStep`

- 目的: 画像を水平・垂直方向に反転
- コンストラクタ: `FlipStep(horizontal=False, vertical=False)`
- メモ: `horizontal` と `vertical` の少なくとも一方を `True` にする必要があります。内部的には OpenCV の `cv2.flip` を利用します。

## サンプルデータで試す

- `python samples/basic_usage.py` を実行すると、サンプル入力に対するパイプラインの動作を確認できます。出力は `samples/output` に保存されます。
- ステップの挙動を変更した際にテスト用アセットを更新したい場合は、`python scripts/generate_test_data.py` で決定的なフィクスチャを再生成できます。

## ヒントと次のステップ

- 小さく始める: まず 1 ステップだけで期待通りの出力を確認し、その後ステップを追加しながら調整すると理解しやすくなります。
- 標準ステップと自作ステップを組み合わせる: `apply(image)` を実装したオブジェクトであれば、OpenCV や NumPy を使った独自ロジックもパイプラインに組み込めます。
- 近い将来、よく使う操作を CLI から実行できるようにする計画があります。リポジトリをウォッチしてアップデートを追ってください。
