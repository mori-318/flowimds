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
    input_path=None,
    output_path="samples/output",
)

result = pipeline.run_on_paths(paths)
for mapping in result.output_mappings:
    print(f"{mapping.input_path} -> {mapping.output_path}")
```

`run_on_paths` は対象ファイルを事前に把握している場合や、複数のディレクトリにまたがる入力をまとめて処理したい場合に有効です。
このメソッドは `Pipeline` インスタンスに設定された `input_path` を使用しないため、`input_path` が設定されている状態で `run_on_paths()` を呼び出すとエラーになります。

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

`run_on_arrays` のみを利用する場合、`input_path` / `output_path` は省略可能です。  
同じインスタンスで後から `run()` を呼ぶ場合は、有効なディレクトリを指す `input_path` と `output_path` の両方を、`run_on_paths()` を呼ぶ場合は `output_path` をそれぞれ設定しておく必要があります。これらが未設定のまま実行すると、入力探索や出力保存のタイミングでエラーになります。

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

## エラーハンドリング

### 失敗した画像の理解

画像の処理が失敗した場合、`result.failed_files` に詳細なエラー情報と共に記録されます：

```python
result = pipeline.run()

if result.failed_count > 0:
    print(f"⚠️  {result.failed_count} 枚の画像が失敗しました:")
    for path in result.failed_files:
        print(f"  - {path}")

    # 失敗した画像を別のパイプラインでリトライ
    if result.failed_files:
        print("よりシンプルなパイプラインでリトライ中...")
        retry_pipeline = fi.Pipeline(
            steps=[fi.ResizeStep((256, 256))],  # 最小限の処理
            output_path="retries",
        )
        retry_result = retry_pipeline.run_on_paths(result.failed_files)
        print(f"回復: {retry_result.processed_count}/{result.failed_count}")
```

**一般的な失敗原因**：

- サポートされていない画像形式（対応形式: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif`）
- 破損した画像ファイル
- 大きな画像によるメモリ不足
- パーミッションエラー（読み書き権限）
- 変換後の無効な画像サイズ

### バリデーションエラーの処理

ステップは初期化時にパラメータを検証し、明確なエラーメッセージを提供します：

```python
# バリデーションエラーとその処理の例
try:
    step = fi.ResizeStep((0, 100))  # 無効: 幅は正の整数必須
except ValueError as e:
    print(f"設定エラー: {e}")
    # 修正: 正の次元を使用
    step = fi.ResizeStep((100, 100))

try:
    step = fi.BinarizeStep(mode="fixed")  # 無効: しきい値が未指定
except ValueError as e:
    print(f"二値化エラー: {e}")
    # 修正: しきい値を指定
    step = fi.BinarizeStep(mode="fixed", threshold=128)

try:
    step = fi.DenoiseStep(mode="median", kernel_size=2)  # 無効: 偶数のカーネルサイズ
except ValueError as e:
    print(f"ノイズ除去エラー: {e}")
    # 修正: 奇数のカーネルサイズを使用
    step = fi.DenoiseStep(mode="median", kernel_size=3)
```

**一般的なバリデーションエラー**：

- `ResizeStep`: 次元は正の整数必須
- `BinarizeStep`: `mode='fixed'` の場合はしきい値必須、しきい値は0〜max_valueの範囲内
- `DenoiseStep`: medianモードではkernel_sizeは奇数かつ3以上必須
- `FlipStep`: horizontal/verticalの少なくとも一方がTrue必須
- `RotateStep`: 角度は任意のfloatだが、極端な値は予期しない結果を生む可能性

### 堅牢なパイプライン設計

エラーを優雅に処理する堅牢なパイプラインを構築：

```python
def robust_pipeline_processing(input_path, output_path, max_retries=2):
    """エラー回復とフォールバック戦略付きで画像を処理"""

    def create_pipeline(complexity="full"):
        if complexity == "full":
            return fi.Pipeline(
                steps=[
                    fi.ResizeStep((512, 512)),
                    fi.GrayscaleStep(),
                    fi.DenoiseStep(mode="median", kernel_size=5),
                    fi.BinarizeStep(mode="otsu"),
                ],
                output_path=output_path,
                log=True,
            )
        elif complexity == "simple":
            return fi.Pipeline(
                steps=[fi.ResizeStep((256, 256)), fi.GrayscaleStep()],
                output_path=output_path,
                log=True,
            )
        else:  # minimal
            return fi.Pipeline(
                steps=[fi.ResizeStep((128, 128))],
                output_path=output_path,
                log=True,
            )

    # まず完全なパイプラインを試行
    result = create_pipeline("full").run()

    # 失敗した画像をよりシンプルなパイプラインでリトライ
    failed_files = result.failed_files.copy()
    retry_attempts = 0

    while failed_files and retry_attempts < max_retries:
        retry_attempts += 1
        complexity = ["simple", "minimal"][retry_attempts - 1]

        print(f"リトライ {retry_attempts}/{max_retries}: {complexity} パイプライン...")
        retry_result = create_pipeline(complexity).run_on_paths(failed_files)

        # 失敗ファイルリストを更新
        newly_failed = set(failed_files) - set(retry_result.output_mappings)
        failed_files = list(newly_failed)

        print(f"  回復: {retry_result.processed_count}, 依然として失敗: {len(failed_files)}")

    # 最終サマリー
    print(f"\n最終結果:")
    print(f"  正常に処理: {result.processed_count}")
    print(f"  リトライで回復: {sum(1 for _ in result.output_mappings) - result.processed_count}")
    print(f"  恒久的に失敗: {len(failed_files)}")

    if failed_files:
        print(f"  失敗ファイル: {failed_files}")

    return result
```

### 失敗した処理のデバッグ

画像の処理が失敗した場合、以下のデバッグ戦略を使用：

```python
def debug_failed_images(failed_files):
    """特定の画像がなぜ失敗したかを分析"""

    for file_path in failed_files:
        print(f"\n{file_path} をデバッグ中:")

        # ファイルが存在し読み取り可能かチェック
        try:
            image = fi.read_image(str(file_path))
            if image is None:
                print("  - 画像を読み取れません（破損の可能性）")
                continue
            print(f"  - 画像形状: {image.shape}")
            print(f"  - 画像データ型: {image.dtype}")
        except Exception as e:
            print(f"  - 読み取りエラー: {e}")
            continue

        # 個別のステップで処理を試行
        test_steps = [
            fi.ResizeStep((256, 256)),
            fi.GrayscaleStep(),
        ]

        for i, step in enumerate(test_steps):
            try:
                processed = step.apply(image)
                print(f"  - ステップ {i+1} ({step.__class__.__name__}): OK")
                image = processed  # 次のステップのために結果を使用
            except Exception as e:
                print(f"  - ステップ {i+1} ({step.__class__.__name__}): 失敗 - {e}")
                break
```

## パイプラインの構成

パスは `str` と `pathlib.Path` のどちらでも指定できます。主な設定項目を下表にまとめます。

| 設定項目 | 型 | 説明 |
| --- | --- | --- |
| `steps` | `PipelineStep` の反復可能オブジェクト | 各画像に順番に適用される変換のリスト。`apply(image)` を持つオブジェクトなら自作ステップも使えます。 |
| `input_path` | `str` または `Path`（任意） | `run` 使用時に画像を探索するディレクトリ。`run_on_arrays` だけ使う場合は省略可能です。 |
| `output_path` | `str` または `Path`（任意） | 変換後のファイルを書き出すディレクトリ。`run` / `run_on_paths` を使うときは必須ですが、`run_on_arrays` のみなら省略できます。 |
| `recursive` | `bool` | 画像収集時にサブディレクトリも走査するかどうか。 |
| `preserve_structure` | `bool` | `True` の場合、入力の階層構造を `output_path` 配下に再現します。`False` ならすべて直下に保存されます。 |
| `worker_count` | `int`（任意） | 並列処理に使用する最大ワーカースレッド数。`None` でCPUコアの約70%、`1` で逐次処理、`0` で全コア使用。 |
| `log` | `bool` | 処理中のプログレスバーと情報ログを有効にします。 |

`steps` の順序はそのまま処理順序になります。前のステップの出力が次のステップの入力として渡される点に注意してください。

## パフォーマンスチューニング

### 並列処理

デフォルトでは、flowimdsは利用可能なCPUコアの約70%を使用して、パフォーマンスとシステム応答性のバランスを取ります：

```python
# ワーカー数を明示的に制御
pipeline = fi.Pipeline(
    steps=[
        fi.ResizeStep((512, 512)),
        fi.GrayscaleStep(),
    ],
    input_path="input",
    output_path="output",
    worker_count=8,  # 8つのワーカースレッドを使用
)
```

**ワーカー数のガイドライン**：
- `worker_count=None`（デフォルト）：CPUコアの約70%を自動検出
- `worker_count=1`：逐次処理（デバッグ時に便利）
- `worker_count=0`：利用可能な全CPUコアを使用

**パフォーマンスのヒント**：
- I/Oバウンドなワークロード（小さい画像が多数）：`worker_count = cpu_count * 1.5` を検討
- CPUバウンドなワークロード（大きい画像、複雑な変換）：`worker_count = cpu_count * 0.7` を使用
- 大きいワーカー数ではメモリ使用量を監視（各ワーカーが画像をメモリに保持）

### 進捗モニタリング

ロギングを有効にして、パイプラインの実行を追跡しリアルタイムのフィードバックを取得：

```python
pipeline = fi.Pipeline(
    steps=[
        fi.ResizeStep((256, 256)),
        fi.DenoiseStep(),
    ],
    input_path="large_dataset",
    output_path="processed",
    log=True,  # プログレスバーとログを有効化
)

result = pipeline.run()
```

`log=True` の場合、以下が表示されます：
- 完了率を示すプログレスバー（tqdm経由）
- 起動時のワーカー/コア数情報
- 長時間実行される操作の定期的な進捗更新

### メモリに関する考慮事項

- 大きな画像は並列処理中により多くのメモリを消費
- 各ワーカースレッドは少なくとも1つの画像をメモリに保持
- メモリエラーが発生した場合、`worker_count` を減らすか、より小さいバッチで画像を処理
- ファイル永続化が不要な場合は、`run_on_arrays` を使用したメモリ効率の良いインメモリ処理を検討

```python
# メモリ効率を考慮したバッチ処理の例
import os
from pathlib import Path

def process_in_batches(input_dir, output_dir, batch_size=100):
    """メモリ使用量を制御するために画像をバッチで処理"""
    all_images = list(Path(input_dir).rglob("*.jpg"))

    for i in range(0, len(all_images), batch_size):
        batch = all_images[i:i + batch_size]
        pipeline = fi.Pipeline(
            steps=[fi.ResizeStep((512, 512))],
            output_path=output_dir,
            worker_count=4,  # メモリに保守的な設定
        )
        result = pipeline.run_on_paths(batch)
        print(f"バッチ {i//batch_size + 1}: {result.processed_count} 枚処理済み")
```

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

- 小さく始めましょう：まず単一ステップで中間出力を検証し、問題がなければ追加します。
- 標準・自作ステップを混在：`apply(image)` を持つオブジェクトなら何でも参加できるため、独自の OpenCV や NumPy ルーチンを簡単にラップできます。
- GitHub リポジトリをウォッチ：今後追加される CLI ツールで、Python コードなしで一般的なパイプライン操作が可能になります。

## トラブルシューティング

### 画像が処理されない

**問題**: `run()` が0枚の処理済み画像を返す

**解決策**:

1. 入力ディレクトリが存在し、画像が含まれているか確認
2. サポート形式を確認: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif`
3. ロギングを有効にして発行詳細を確認:

```python
pipeline = fi.Pipeline(..., log=True)
result = pipeline.run()
```

4. 画像がサブディレクトリにある場合は `recursive=True` を試す

5. ディレクトリ権限を確認:

```python
import os
input_dir = "/path/to/input"
print(f"ディレクトリ存在: {os.path.exists(input_dir)}")
print(f"ディレクトリ読取可能: {os.access(input_dir, os.R_OK)}")

# ディレクトリ内の画像ファイルを一覧
from pathlib import Path
image_files = list(Path(input_dir).rglob("*"))
supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
images = [f for f in image_files if f.suffix.lower() in supported_extensions]
print(f"サポート対象画像 {len(images)} 枚を発見")
```

### 出力ファイルが作成されない

**問題**: パイプラインは完了するが出力ファイルが表示されない

**解決策**:
1. `output_path` ディレクトリが存在するか確認（自動作成）
2. 出力ディレクトリの書き込み権限を確認
3. 特定の失敗について `result.failed_files` を検査
4. 出力マッピングを確認:

```python
for mapping in result.output_mappings:
    print(f"{mapping.input_path} -> {mapping.output_path}")

# 出力ファイルが実際に存在するか確認
import os
for mapping in result.output_mappings:
    if os.path.exists(mapping.output_path):
        print(f"✓ {mapping.output_path}")
    else:
        print(f"✗ 欠落: {mapping.output_path}")
```

### ファイル名の衝突（フラット化出力）

**問題**: `preserve_structure=False` 時にファイルが上書きされる

**動作**: flowimdsは重複に自動的に `_no{N}` 接尾辞を追加:
- `image.png` → `image.png`
- `image.png`（重複）→ `image_no2.png`
- `image.png`（重複）→ `image_no3.png`

**例**:
```python
# 衝突処理のデモ
pipeline = fi.Pipeline(
    steps=[fi.GrayscaleStep()],
    input_path="input",  # 含む: folder1/image.png, folder2/image.png
    output_path="output",
    preserve_structure=False,  # 出力をフラット化
    log=True,
)

result = pipeline.run()
for mapping in result.output_mappings:
    print(f"{mapping.input_path} -> {mapping.output_path}")
# 出力:
# input/folder1/image.png -> output/image.png
# input/folder2/image.png -> output/image_no2.png
```

### パフォーマンスの問題

**問題**: 処理が予想より遅い

**診断**:
```python
import time
import psutil

def profile_pipeline(pipeline):
    """パイプラインのパフォーマンスとリソース使用量をプロファイル"""

    # システムリソースを監視
    cpu_before = psutil.cpu_percent()
    memory_before = psutil.virtual_memory().percent

    start_time = time.time()
    result = pipeline.run()
    end_time = time.time()

    cpu_after = psutil.cpu_percent()
    memory_after = psutil.virtual_memory().percent

    print(f"パフォーマンス指標:")
    print(f"  実行時間: {end_time - start_time:.2f} 秒")
    print(f"  処理済み画像: {result.processed_count}")
    print(f"  画像/秒: {result.processed_count / (end_time - start_time):.2f}")
    print(f"  CPU使用率: {cpu_before:.1f}% -> {cpu_after:.1f}%")
    print(f"  メモリ使用率: {memory_before:.1f}% -> {memory_after:.1f}%")

    return result
```

**最適化戦略**:
1. I/Oバウンドなワークロード（小さい画像が多数）: `worker_count` を増加
2. CPUバウンドなワークロード（大きい画像、複雑な変換）: `worker_count` を減少
3. メモリ使用量を監視し、必要に応じてワーカーを減少
4. 非常に大きなコレクションの場合はバッチ処理を検討
5. ファイル永続化が不要な場合は `run_on_arrays` を使用

### メモリエラー

**問題**: `MemoryError` またはシステムが応答しなくなる

**解決策**:
```python
def memory_safe_processing(input_path, output_path):
    """メモリ制約を考慮して画像を処理"""

    # 保守的な設定で開始
    pipeline = fi.Pipeline(
        steps=[fi.ResizeStep((512, 512))],
        input_path=input_path,
        output_path=output_path,
        worker_count=1,  # 逐次処理
        log=True,
    )

    try:
        result = pipeline.run()
        return result
    except MemoryError:
        print("メモリエラーが発生、バッチ処理を試行中...")

        # より小さいバッチで処理
        from pathlib import Path
        all_images = list(Path(input_path).rglob("*.jpg"))
        batch_size = 50  # 利用可能なメモリに応じて調整

        total_processed = 0
        for i in range(0, len(all_images), batch_size):
            batch = all_images[i:i + batch_size]
            batch_pipeline = fi.Pipeline(
                steps=[fi.ResizeStep((512, 512))],
                output_path=output_path,
                worker_count=1,
            )
            batch_result = batch_pipeline.run_on_paths(batch)
            total_processed += batch_result.processed_count
            print(f"バッチ {i//batch_size + 1}: {batch_result.processed_count} 枚処理済み")

        return fi.PipelineResult(
            processed_count=total_processed,
            failed_count=0,
            failed_files=[],
            output_mappings=[],
            duration_seconds=0,
            settings={},
        )
```

### 日本語のファイル名とパス

**注**: flowimdsは非ASCIIパスのためのOpenCVの特別な処理を使用します。ファイル名とパスの日本語文字は完全にサポートされています。

```python
# 日本語ファイル名は正しく動作
pipeline = fi.Pipeline(
    steps=[fi.ResizeStep((256, 256))],
    input_path="写真/入力",  # 日本語ディレクトリ名
    output_path="写真/出力",  # 日本語ディレクトリ名
    recursive=True,
    log=True,
)

result = pipeline.run()
print(f"日本語パスで {result.processed_count} 枚の画像を処理")
```

### ステップ固有の問題

**ResizeStepが予期しない結果を生成**:
```python
# リサイズ前後の画像次元を確認
def debug_resize_step():
    test_image = np.random.randint(0, 255, (1000, 800, 3), dtype=np.uint8)
    print(f"元の形状: {test_image.shape}")

    resize_step = fi.ResizeStep((512, 512))
    resized = resize_step.apply(test_image)
    print(f"リサイズ後の形状: {resized.shape}")

    # 注: OpenCVは (幅, 高さ) 規約を使用
    # したがって (512, 512) は512x512出力を生成
```

**BinarizeStepが期待通りに動作しない**:
```python
# BinarizeStepは常に最初にグレースケールに変換
def debug_binarize_step():
    # カラーテスト画像を作成
    color_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    print(f"入力形状: {color_image.shape}, チャンネル: {color_image.shape[2] if len(color_image.shape) == 3 else 1}")

    binarize_step = fi.BinarizeStep(mode="otsu")
    result = binarize_step.apply(color_image)
    print(f"出力形状: {result.shape}, チャンネル: {result.shape[2] if len(result.shape) == 3 else 1}")
    print(f"出力データ型: {result.dtype}")
    print(f"一意の値: {np.unique(result)}")
```

### ヘルプの取得

ここでカバーされていない問題に遭遇した場合:

1. **ロギングを有効に**して詳細な実行情報を取得
2. **GitHubリポジトリを確認**して既知の問題とディスカッションを確認
3. **最小再現可能な例を作成**してバグを報告
4. **システム情報を含める**（Pythonバージョン、OS、メモリ）
5. **問題を再現するサンプル画像を提供**（可能であれば）

```python
# バグレポート用テンプレート
def create_bug_report():
    """バグレポートに有用な情報を生成"""

    import sys
    import platform
    import cv2
    import numpy as np

    print("システム情報:")
    print(f"  Python: {sys.version}")
    print(f"  プラットフォーム: {platform.platform()}")
    print(f"  OpenCV: {cv2.__version__}")
    print(f"  NumPy: {np.__version__}")
    print(f"  flowimds: {fi.__version__ if hasattr(fi, '__version__') else 'unknown'}")

    # メモリ情報
    import psutil
    memory = psutil.virtual_memory()
    print(f"  総RAM: {memory.total / (1024**3):.1f} GB")
    print(f"  利用可能RAM: {memory.available / (1024**3):.1f} GB")
```

## 実践的なレシピ集

### レシピ1: Web用バッチ画像リサイズ

アスペクト比を維持しながら、ディレクトリ内の画像を複数のWeb対応サイズにリサイズ：

```python
def resize_for_web(input_dir, output_dir, sizes=[(1920, 1080), (1280, 720), (640, 360)]):
    """画像を複数のWeb対応フォーマットにリサイズ"""

    for width, height in sizes:
        print(f"{width}x{height} サイズを処理中...")

        pipeline = fi.Pipeline(
            steps=[fi.ResizeStep((width, height))],
            input_path=input_dir,
            output_path=f"{output_dir}/{width}x{height}",
            log=True,
            worker_count=4,
        )

        result = pipeline.run()
        print(f"  完了: {result.processed_count} 枚の画像")
        if result.failed_count > 0:
            print(f"  失敗: {result.failed_count} 枚の画像")

# 使用例
resize_for_web("photos/original", "photos/web")
```

### レシピ2: 文書スキャンパイプライン

スキャンした文書を傾き補正、ノイズ除去、二値化で処理：

```python
def document_preprocessing(input_dir, output_dir):
    """OCRまたはアーカイブ用にスキャン文書を準備"""

    pipeline = fi.Pipeline(
        steps=[
            fi.ResizeStep((2000, 3000)),  # サイズを標準化
            fi.GrayscaleStep(),           # グレースケールに変換
            fi.DenoiseStep(mode="gaussian", kernel_size=5),  # ノイズを除去
            fi.BinarizeStep(mode="otsu"),  # 最適なしきい値処理
        ],
        input_path=input_dir,
        output_path=output_dir,
        log=True,
        worker_count=2,  # 大きな文書には保守的な設定
    )

    result = pipeline.run()
    print(f"文書処理完了:")
    print(f"  正常に処理: {result.processed_count}")
    print(f"  失敗: {result.failed_count}")

    return result

# 使用例
document_preprocessing("scans/input", "scans/processed")
```

### レシピ3: 機械学習データ準備

一貫した前処理でMLトレーニング用の画像データセットを準備：

```python
def prepare_ml_dataset(input_dir, output_dir, target_size=(224, 224), augment=False):
    """機械学習トレーニング用の画像を準備"""

    steps = [fi.ResizeStep(target_size)]

    if augment:
        # データ拡張ステップを追加
        steps.extend([
            fi.RandomRotationStep(angle_range=(-15, 15)),
            fi.RandomFlipStep(horizontal=True, vertical=False),
        ])

    pipeline = fi.Pipeline(
        steps=steps,
        input_path=input_dir,
        output_path=output_dir,
        preserve_structure=True,  # クラスフォルダを維持
        log=True,
        worker_count=6,
    )

    result = pipeline.run()

    # データセット統計を生成
    print(f"データセット準備完了:")
    print(f"  総画像数: {result.processed_count}")
    print(f"  処理時間: {result.duration_seconds:.2f} 秒")
    print(f"  画像あたりの平均時間: {result.duration_seconds/result.processed_count:.3f} 秒")

    return result

# トレーニングデータでの使用例
prepare_ml_dataset("dataset/raw/train", "dataset/processed/train", augment=True)
prepare_ml_dataset("dataset/raw/val", "dataset/processed/val", augment=False)
```

### レシピ4: ウォーターマーク付きサムネイル生成

画像ギャラリー用に自動ウォーターマーク付きサムネイルを作成：

```python
class WatermarkStep:
    """画像にウォーターマークを追加するカスタムステップ"""

    def __init__(self, watermark_text="© My Gallery", opacity=0.7):
        self.watermark_text = watermark_text
        self.opacity = opacity

    def apply(self, image):
        """半透明テキストウォーターマークを追加"""
        import cv2

        # OpenCVテキスト操作用にBGRに変換
        if len(image.shape) == 2:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = image.copy()

        # ウォーターマークテキストを追加
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2

        # 位置決めのためにテキストサイズを取得
        (text_width, text_height), baseline = cv2.getTextSize(
            self.watermark_text, font, font_scale, thickness
        )

        # 右下隅にウォーターマークを配置
        h, w = image_bgr.shape[:2]
        x = w - text_width - 20
        y = h - text_height - 20

        # 不透明度でテキストを追加
        overlay = image_bgr.copy()
        cv2.putText(overlay, self.watermark_text, (x, y),
                   font, font_scale, (255, 255, 255), thickness)

        # 元の画像とブレンド
        result = cv2.addWeighted(overlay, self.opacity, image_bgr, 1 - self.opacity, 0)

        return result

def create_thumbnails_with_watermark(input_dir, output_dir, thumbnail_size=(300, 300)):
    """ギャラリー表示用にウォーターマーク付きサムネイルを生成"""

    pipeline = fi.Pipeline(
        steps=[
            fi.ResizeStep(thumbnail_size),
            WatermarkStep("© My Gallery 2024"),
        ],
        input_path=input_dir,
        output_path=output_dir,
        log=True,
        worker_count=8,
    )

    result = pipeline.run()
    print(f"サムネイル生成完了: {result.processed_count} 個のサムネイルを作成")

    return result

# 使用例
create_thumbnails_with_watermark("gallery/full_size", "gallery/thumbnails")
```

### レシピ5: 医療画像前処理

コントラスト強調とノイズ除去で医療画像を標準化：

```python
def medical_image_preprocessing(input_dir, output_dir):
    """コントラスト強調で医療画像を前処理"""

    class ContrastEnhancementStep:
        """CLAHEを使用する医療画像コントラスト強調用カスタムステップ"""

        def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
            self.clip_limit = clip_limit
            self.tile_grid_size = tile_grid_size

        def apply(self, image):
            """CLAHE（制限付き適応ヒストグラム平坦化）を適用"""
            import cv2

            # 必要に応じてグレースケールに変換
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # CLAHEを適用
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit,
                                   tileGridSize=self.tile_grid_size)
            enhanced = clahe.apply(gray)

            return enhanced

    pipeline = fi.Pipeline(
        steps=[
            fi.ResizeStep((512, 512)),  # 医療画像サイズを標準化
            ContrastEnhancementStep(clip_limit=3.0),  # コントラストを強調
            fi.DenoiseStep(mode="median", kernel_size=3),  # ノイズを除去
        ],
        input_path=input_dir,
        output_path=output_dir,
        log=True,
        worker_count=2,  # 医療画像には保守的な設定
    )

    result = pipeline.run()
    print(f"医療画像前処理完了:")
    print(f"  処理済み: {result.processed_count} 枚の画像")
    print(f"  失敗: {result.failed_count} 枚の画像")

    return result

# 使用例
medical_image_preprocessing("medical/raw", "medical/processed")
```

### レシピ6: リアルタイム処理パイプライン

ディレクトリに追加された画像を処理（モニタリングに有用）：

```python
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ImageProcessorHandler(FileSystemEventHandler):
    """ディレクトリに表示される新しい画像を処理"""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.pipeline = fi.Pipeline(
            steps=[
                fi.ResizeStep((1024, 768)),
                fi.GrayscaleStep(),
                fi.DenoiseStep(mode="gaussian", kernel_size=3),
            ],
            output_path=output_dir,
            worker_count=2,
        )

    def on_created(self, event):
        """新しいファイル作成を処理"""
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        if file_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}:
            print(f"新しい画像を処理中: {file_path.name}")

            # ファイルが完全に書き込まれるまで少し待機
            time.sleep(1)

            try:
                result = self.pipeline.run_on_paths([file_path])
                if result.processed_count > 0:
                    print(f"  ✓ 正常に処理完了")
                else:
                    print(f"  ✗ 処理失敗")
            except Exception as e:
                print(f"  ✗ エラー: {e}")

def monitor_directory(input_dir, output_dir):
    """入力ディレクトリを監視し、新しい画像を自動処理"""

    event_handler = ImageProcessorHandler(output_dir)
    observer = Observer()
    observer.schedule(event_handler, input_dir, recursive=True)
    observer.start()

    print(f"{input_dir} の新しい画像を監視中...")
    print(f"処理済み画像は {output_dir} に保存されます")
    print("監視を停止するには Ctrl+C を押してください")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n監視を停止しました")

    observer.join()

# 使用例
# monitor_directory("watch/input", "watch/output")
```

### レシピ7: 品質評価パイプライン

画像品質を評価し、低品質画像をフィルタリング：

```python
def assess_image_quality(image):
    """様々な指標で画像品質を評価"""
    import cv2

    # 必要に応じてグレースケールに変換
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # ラプラシアン分散でシャープネスを計算
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 輝度を計算
    brightness = gray.mean()

    # コントラスト（標準偏差）を計算
    contrast = gray.std()

    return {
        'sharpness': laplacian_var,
        'brightness': brightness,
        'contrast': contrast,
    }

class QualityFilterStep:
    """品質指標に基づいて画像をフィルタリングするカスタムステップ"""

    def __init__(self, min_sharpness=100, min_contrast=30):
        self.min_sharpness = min_sharpness
        self.min_contrast = min_contrast
        self.quality_scores = []

    def apply(self, image):
        """品質を評価し、基準を満たす場合に画像を返す"""
        quality = assess_image_quality(image)
        self.quality_scores.append(quality)

        # 品質しきい値をチェック
        if (quality['sharpness'] >= self.min_sharpness and
            quality['contrast'] >= self.min_contrast):
            return image
        else:
            # 画像をフィルタリングする必要があることを示すためにNoneを返す
            # 注: これを処理するにはカスタムパイプラインロジックが必要
            return image

def quality_assessment_pipeline(input_dir, output_dir, good_dir, poor_dir):
    """品質評価に基づいて画像を分離"""

    quality_step = QualityFilterStep(min_sharpness=100, min_contrast=30)

    pipeline = fi.Pipeline(
        steps=[
            fi.ResizeStep((1024, 768)),
            quality_step,
        ],
        input_path=input_dir,
        output_path=output_dir,
        log=True,
        worker_count=4,
    )

    result = pipeline.run()

    # 品質スコアを分析
    if quality_step.quality_scores:
        avg_sharpness = sum(q['sharpness'] for q in quality_step.quality_scores) / len(quality_step.quality_scores)
        avg_contrast = sum(q['contrast'] for q in quality_step.quality_scores) / len(quality_step.quality_scores)

        print(f"品質評価完了:")
        print(f"  処理済み画像: {result.processed_count}")
        print(f"  平均シャープネス: {avg_sharpness:.2f}")
        print(f"  平均コントラスト: {avg_contrast:.2f}")

    return result, quality_step.quality_scores

# 使用例
result, scores = quality_assessment_pipeline("photos/all", "photos/processed", "photos/good", "photos/poor")
```

## カスタムステップの作成

### PipelineStepプロトコルの理解

flowimdsのすべてのパイプラインステップは`PipelineStep`プロトコルに従います。これには単一のメソッドが必要です：

```python
from flowimds.steps.base import PipelineStep
import numpy as np

class MyCustomStep:
    """PipelineStepプロトコルを実装するカスタムステップ"""

    def apply(self, image: np.ndarray) -> np.ndarray:
        """提供された画像を変換し結果を返す

        Args:
            image: numpy配列としての入力画像（2Dまたは3D）

        Returns:
            同じまたは異なる次元のnumpy配列としての変換された画像
        """
        # カスタム処理ロジックをここに記述
        return processed_image
```

### 基本的なカスタムステップの例

#### 例1: シンプルな輝度調整

```python
class BrightnessStep:
    """定数値を加えて画像の輝度を調整"""

    def __init__(self, brightness_factor: float = 1.0):
        """輝度調整を初期化

        Args:
            brightness_factor: ピクセル値の乗数（1.0 = 変化なし）
        """
        self.brightness_factor = brightness_factor

    def apply(self, image: np.ndarray) -> np.ndarray:
        """輝度調整を適用"""
        # オーバーフローを避けるためにfloatに変換
        adjusted = image.astype(np.float32) * self.brightness_factor

        # 有効範囲に値をクリップして戻す
        adjusted = np.clip(adjusted, 0, 255)
        return adjusted.astype(image.dtype)

# 使用例
pipeline = fi.Pipeline(
    steps=[
        fi.ResizeStep((512, 512)),
        BrightnessStep(brightness_factor=1.2),  # 輝度を20%増加
        fi.GrayscaleStep(),
    ],
    input_path="input",
    output_path="output",
)
```

#### 例2: カスタムパラメータ付きガウシアンぼかし

```python
import cv2

class CustomBlurStep:
    """設定可能なパラメータでガウシアンぼかしを適用"""

    def __init__(self, kernel_size: int = 5, sigma_x: float = 1.0):
        """ぼかしステップを初期化

        Args:
            kernel_size: ガウシアンカーネルのサイズ（奇数である必要あり）
            sigma_x: X方向の標準偏差
        """
        if kernel_size % 2 == 0:
            raise ValueError("kernel_sizeは奇数である必要があります")
        self.kernel_size = kernel_size
        self.sigma_x = sigma_x

    def apply(self, image: np.ndarray) -> np.ndarray:
        """ガウシアンぼかしを適用"""
        return cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), self.sigma_x)

# 使用例
pipeline = fi.Pipeline(
    steps=[
        fi.ResizeStep((1024, 768)),
        CustomBlurStep(kernel_size=7, sigma_x=2.0),
        fi.BinarizeStep(mode="otsu"),
    ],
)
```

### 高度なカスタムステップの例

#### 例3: 複数アルゴリズムによるエッジ検出

```python
class EdgeDetectionStep:
    """アルゴリズム選択付きエッジ検出"""

    def __init__(self, method: str = "canny", **kwargs):
        """エッジ検出を初期化

        Args:
            method: エッジ検出メソッド（'canny'、'sobel'、'laplacian'）
            **kwargs: メソッド固有のパラメータ
        """
        self.method = method.lower()
        self.kwargs = kwargs

        if self.method not in {"canny", "sobel", "laplacian"}:
            raise ValueError("methodは'canny'、'sobel'、'laplacian'のいずれかである必要があります")

    def apply(self, image: np.ndarray) -> np.ndarray:
        """エッジ検出を適用"""
        # 必要に応じてグレースケールに変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if self.method == "canny":
            low_threshold = self.kwargs.get("low_threshold", 50)
            high_threshold = self.kwargs.get("high_threshold", 150)
            return cv2.Canny(gray, low_threshold, high_threshold)

        elif self.method == "sobel":
            ksize = self.kwargs.get("ksize", 3)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
            return np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)

        elif self.method == "laplacian":
            ksize = self.kwargs.get("ksize", 3)
            return cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize).astype(np.uint8)

# 使用例
pipeline = fi.Pipeline(
    steps=[
        fi.ResizeStep((512, 512)),
        fi.GrayscaleStep(),
        EdgeDetectionStep(method="canny", low_threshold=100, high_threshold=200),
    ],
)
```

#### 例4: ヒストグラム平坦化

```python
class HistogramEqualizationStep:
    """オプションのCLAHEでヒストグラム平坦化を適用"""

    def __init__(self, use_clahe: bool = False, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        """ヒストグラム平坦化を初期化

        Args:
            use_clahe: 標準ヒストグラム平坦化の代わりにCLAHEを使用
            clip_limit: CLAHEのコントラスト制限しきい値
            tile_grid_size: タイルベースヒストグラム平坦化のグリッドサイズ
        """
        self.use_clahe = use_clahe
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def apply(self, image: np.ndarray) -> np.ndarray:
        """ヒストグラム平坦化を適用"""
        # 必要に応じてグレースケールに変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            return clahe.apply(gray)
        else:
            return cv2.equalizeHist(gray)

# 使用例
pipeline = fi.Pipeline(
    steps=[
        fi.ResizeStep((512, 512)),
        HistogramEqualizationStep(use_clahe=True, clip_limit=3.0),
    ],
)
```

### 状態を持つカスタムステップ

#### 例5: 画像統計に基づく適応しきい値

```python
class AdaptiveThresholdStep:
    """画像統計に基づく適応しきい値処理"""

    def __init__(self, target_mean: float = 128.0, max_iterations: int = 10):
        """適応しきい値を初期化

        Args:
            target_mean: しきい値処理後の目標平均ピクセル値
            max_iterations: 最適しきい値を見つける最大反復回数
        """
        self.target_mean = target_mean
        self.max_iterations = max_iterations
        self.final_threshold = None  # 使用されたしきい値を保存

    def apply(self, image: np.ndarray) -> np.ndarray:
        """適応しきい値処理を適用"""
        # 必要に応じてグレースケールに変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 最適しきい値の二分探索
        low, high = 0, 255
        best_threshold = 127

        for _ in range(self.max_iterations):
            threshold = (low + high) // 2
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            current_mean = binary.mean()

            if abs(current_mean - self.target_mean) < 1.0:
                best_threshold = threshold
                break
            elif current_mean < self.target_mean:
                high = threshold
            else:
                low = threshold
            best_threshold = threshold

        self.final_threshold = best_threshold
        _, result = cv2.threshold(gray, best_threshold, 255, cv2.THRESH_BINARY)
        return result

# しきい値検査付き使用例
threshold_step = AdaptiveThresholdStep(target_mean=100.0)
pipeline = fi.Pipeline(
    steps=[
        fi.ResizeStep((512, 512)),
        fi.GrayscaleStep(),
        threshold_step,
    ],
)

result = pipeline.run()
print(f"使用された最終しきい値: {threshold_step.final_threshold}")
```

### カスタムステップのベストプラクティス

#### 1. 入力検証

```python
class RobustCustomStep:
    """包括的な入力検証付きカスタムステップ"""

    def __init__(self, param1: float, param2: int):
        """パラメータ検証で初期化"""
        if not isinstance(param1, (int, float)):
            raise TypeError("param1は数値である必要があります")
        if not 0 <= param1 <= 1.0:
            raise ValueError("param1は0から1の間である必要があります")
        if not isinstance(param2, int) or param2 <= 0:
            raise ValueError("param2は正の整数である必要があります")

        self.param1 = float(param1)
        self.param2 = param2

    def apply(self, image: np.ndarray) -> np.ndarray:
        """画像検証で適用"""
        if not isinstance(image, np.ndarray):
            raise TypeError("入力はnumpy配列である必要があります")
        if image.size == 0:
            raise ValueError("入力配列は空であってはなりません")
        if len(image.shape) not in {2, 3}:
            raise ValueError("入力は2Dまたは3D配列である必要があります")

        # 処理ロジックをここに記述
        return processed_image
```

#### 2. メモリ効率

```python
class MemoryEfficientStep:
    """メモリ効率のためにチャンクで画像を処理するカスタムステップ"""

    def __init__(self, chunk_size: int = 1024):
        """処理用チャンクサイズで初期化"""
        self.chunk_size = chunk_size

    def apply(self, image: np.ndarray) -> np.ndarray:
        """メモリを節約するためにチャンクで画像を処理"""
        height, width = image.shape[:2]
        result = np.zeros_like(image)

        # 水平チャンクで処理
        for y in range(0, height, self.chunk_size):
            y_end = min(y + self.chunk_size, height)
            chunk = image[y:y_end]

            # チャンクを処理
            processed_chunk = self._process_chunk(chunk)
            result[y:y_end] = processed_chunk

        return result

    def _process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """画像の単一チャンクを処理"""
        # チャンク固有の処理ロジック
        return chunk
```

#### 3. エラーハンドリングとロギング

```python
import logging

class LoggingStep:
    """組み込みロギングとエラーハンドリング付きカスタムステップ"""

    def __init__(self, operation_name: str = "custom_operation"):
        """ロギング設定で初期化"""
        self.operation_name = operation_name
        self.logger = logging.getLogger(f"flowimds.{operation_name}")

    def apply(self, image: np.ndarray) -> np.ndarray:
        """包括的なエラーハンドリングで適用"""
        try:
            self.logger.debug(f"形状 {image.shape} の画像を処理中")

            # 入力を検証
            if image is None or image.size == 0:
                raise ValueError("無効な入力画像")

            # 画像を処理
            result = self._process_image(image)

            self.logger.debug(f"画像の処理に成功、出力形状 {result.shape}")
            return result

        except Exception as e:
            self.logger.error(f"{self.operation_name} でのエラー: {e}")
            # より多くのコンテキストで再発生
            raise RuntimeError(f"{self.operation_name} での画像処理に失敗: {e}") from e

    def _process_image(self, image: np.ndarray) -> np.ndarray:
        """実際の処理ロジック"""
        # 処理コードをここに記述
        return image
```

### カスタムステップのテスト

```python
import unittest
import numpy as np

class TestCustomStep(unittest.TestCase):
    """カスタムステップのテストスイート"""

    def setUp(self):
        """テストフィクスチャを設定"""
        self.test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.test_gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

    def test_brightness_step(self):
        """BrightnessStep機能をテスト"""
        step = BrightnessStep(brightness_factor=1.5)

        # カラー画像でテスト
        result = step.apply(self.test_image)
        self.assertEqual(result.shape, self.test_image.shape)
        self.assertTrue(np.all(result >= self.test_image))  # より明るくなるはず

        # グレースケール画像でテスト
        result_gray = step.apply(self.test_gray)
        self.assertEqual(result_gray.shape, self.test_gray.shape)

    def test_invalid_parameters(self):
        """パラメータ検証をテスト"""
        with self.assertRaises(ValueError):
            BrightnessStep(brightness_factor=-1.0)

    def test_edge_detection_methods(self):
        """異なるエッジ検出メソッドをテスト"""
        for method in ["canny", "sobel", "laplacian"]:
            step = EdgeDetectionStep(method=method)
            result = step.apply(self.test_image)
            self.assertEqual(len(result.shape), 2)  # グレースケールになるはず
            self.assertTrue(np.all(result >= 0))  # 負でないはず

# テストを実行
if __name__ == "__main__":
    unittest.main()
```

### カスタムステップの統合

```python
# 組み込みとカスタムステップを混在させたパイプラインを作成
custom_pipeline = fi.Pipeline(
    steps=[
        fi.ResizeStep((512, 512)),
        BrightnessStep(brightness_factor=1.2),
        CustomBlurStep(kernel_size=5, sigma_x=1.0),
        EdgeDetectionStep(method="canny", low_threshold=50, high_threshold=150),
        HistogramEqualizationStep(use_clahe=True),
    ],
    input_path="input",
    output_path="output",
    log=True,
    worker_count=4,
)

# パイプラインを実行
result = custom_pipeline.run()
print(f"カスタムパイプラインで {result.processed_count} 枚の画像を処理")
```

## APIリファレンス

### コアクラス

#### Pipeline

画像処理パイプラインを作成・実行するためのメインクラス。

```python
class Pipeline:
    """並列実行サポート付き画像処理パイプライン"""

    def __init__(
        self,
        steps: List[PipelineStep],
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        worker_count: int = 4,
        preserve_structure: bool = True,
        log: bool = False,
    ):
        """新しいパイプラインを初期化

        Args:
            steps: PipelineStepプロトコルを実装する処理ステップのリスト
            input_path: 入力画像を含むディレクトリ（オプション）
            output_path: 出力画像用のディレクトリ（オプション）
            worker_count: 並列ワーカー数（デフォルト: 4）
            preserve_structure: ディレクトリ構造を維持するか（デフォルト: True）
            log: 詳細なロギングを有効にする（デフォルト: False）
        """
```

**メソッド:**

- `run() -> PipelineResult`: ディレクトリでパイプラインを実行
- `run_on_paths(paths: List[Path]) -> PipelineResult`: 特定のファイルパスで実行
- `run_on_arrays(images: List[np.ndarray]) -> PipelineResult`: numpy配列で実行

**例:**
```python
pipeline = fi.Pipeline(
    steps=[fi.ResizeStep((512, 512)), fi.GrayscaleStep()],
    input_path="input",
    output_path="output",
    worker_count=8,
    log=True,
)

# ディレクトリで実行
result = pipeline.run()

# 特定のファイルで実行
from pathlib import Path
specific_files = [Path("img1.jpg"), Path("img2.jpg")]
result = pipeline.run_on_paths(specific_files)

# numpy配列で実行
import numpy as np
arrays = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)]
result = pipeline.run_on_arrays(arrays)
```

#### PipelineResult

パイプライン実行の結果とメタデータを含む。

```python
@dataclass
class PipelineResult:
    """パイプライン実行の結果"""

    processed_count: int
    failed_count: int
    failed_files: List[str]
    output_mappings: List[FileMapping]
    duration_seconds: float
    settings: Dict[str, Any]

    @property
    def success_rate(self) -> float:
        """成功率をパーセンテージで計算"""
        if self.processed_count + self.failed_count == 0:
            return 0.0
        return (self.processed_count / (self.processed_count + self.failed_count)) * 100
```

**FileMapping:**
```python
@dataclass
class FileMapping:
    """入力ファイルと出力ファイルのマッピング"""

    input_path: str
    output_path: str
    success: bool
    error_message: Optional[str] = None
```

### 組み込みステップ

#### ResizeStep

画像を指定された寸法にリサイズ。

```python
class ResizeStep:
    """画像を指定された寸法にリサイズ"""

    def __init__(self, size: Tuple[int, int]):
        """リサイズステップを初期化

        Args:
            size: (幅, 高さ) タプルとしての目標サイズ
        """

    def apply(self, image: np.ndarray) -> np.ndarray:
        """画像を目標寸法にリサイズ"""
```

**例:**
```python
# 正方形にリサイズ
step = fi.ResizeStep((512, 512))

# 長方形にリサイズ
step = fi.ResizeStep((1024, 768))

# パイプラインで使用
pipeline = fi.Pipeline(
    steps=[fi.ResizeStep((800, 600))],
    input_path="input",
    output_path="output",
)
```

#### GrayscaleStep

画像をグレースケールに変換。

```python
class GrayscaleStep:
    """カラー画像をグレースケールに変換"""

    def apply(self, image: np.ndarray) -> np.ndarray:
        """画像をグレースケールに変換"""
```

**例:**
```python
step = fi.GrayscaleStep()
result = step.apply(color_image)  # 2D配列を返す
```

#### DenoiseStep

画像にノイズ除去を適用。

```python
class DenoiseStep:
    """画像にノイズ除去を適用"""

    def __init__(self, mode: str = "gaussian", kernel_size: int = 5):
        """ノイズ除去ステップを初期化

        Args:
            mode: ノイズ除去方法（"gaussian", "median", "bilateral"）
            kernel_size: ノイズ除去カーネルのサイズ
        """

    def apply(self, image: np.ndarray) -> np.ndarray:
        """画像にノイズ除去を適用"""
```

**例:**
```python
# ガウシアンぼかし
gaussian_step = fi.DenoiseStep(mode="gaussian", kernel_size=5)

# メディアンフィルタ
median_step = fi.DenoiseStep(mode="median", kernel_size=3)

# バイラテラルフィルタ（エッジを保持）
bilateral_step = fi.DenoiseStep(mode="bilateral")
```

#### BinarizeStep

画像を二値（白黒）に変換。

```python
class BinarizeStep:
    """しきい値処理を使用して画像を二値化"""

    def __init__(self, mode: str = "otsu", threshold: Optional[int] = None):
        """二値化ステップを初期化

        Args:
            mode: しきい値処理方法（"otsu", "adaptive", "fixed"）
            threshold: 固定しきい値（"fixed"モードで必須）
        """

    def apply(self, image: np.ndarray) -> np.ndarray:
        """画像を二値化"""
```

**例:**
```python
# 大津の自動しきい値処理
otsu_step = fi.BinarizeStep(mode="otsu")

# 固定しきい値
fixed_step = fi.BinarizeStep(mode="fixed", threshold=127)

# 適応しきい値処理
adaptive_step = fi.BinarizeStep(mode="adaptive")
```

#### FlipStep

画像を水平・垂直に反転。

```python
class FlipStep:
    """画像を水平・垂直に反転"""

    def __init__(self, horizontal: bool = False, vertical: bool = False):
        """反転ステップを初期化

        Args:
            horizontal: 水平に反転（デフォルト: False）
            vertical: 垂直に反転（デフォルト: False）
        """

    def apply(self, image: np.ndarray) -> np.ndarray:
        """指定された軸に従って画像を反転"""
```

**例:**
```python
# 水平反転
h_flip = fi.FlipStep(horizontal=True)

# 垂直反転
v_flip = fi.FlipStep(vertical=True)

# 水平・垂直両方
both_flip = fi.FlipStep(horizontal=True, vertical=True)
```

### ユーティリティ関数

#### 画像読み込みと保存

```python
def load_image(path: str) -> np.ndarray:
    """ファイルパスから画像を読み込み

    Args:
        path: 画像ファイルへのパス

    Returns:
        numpy配列としての画像

    Raises:
        FileNotFoundError: ファイルが存在しない場合
        ValueError: ファイルが有効な画像でない場合
    """

def save_image(image: np.ndarray, path: str) -> None:
    """画像をファイルパスに保存

    Args:
        image: numpy配列としての画像
        path: 出力ファイルパス

    Raises:
        ValueError: 画像フォーマットがサポートされていない場合
    """
```

#### 画像検証

```python
def validate_image(image: np.ndarray) -> bool:
    """配列が有効な画像か検証

    Args:
        image: 検証する配列

    Returns:
        有効な画像の場合はTrue、そうでなければFalse
    """

def get_image_info(image: np.ndarray) -> Dict[str, Any]:
    """画像配列の情報を取得

    Args:
        image: 画像配列

    Returns:
        画像メタデータを含む辞書
    """
```

### 設定

#### パイプライン設定

```python
class PipelineSettings:
    """パイプライン実行の設定"""

    def __init__(
        self,
        max_workers: int = 4,
        chunk_size: int = 100,
        timeout_seconds: int = 300,
        retry_attempts: int = 3,
        memory_limit_mb: int = 1024,
    ):
        """パイプライン設定を初期化

        Args:
            max_workers: 最大ワーカープロセス数
            chunk_size: 処理チャンクあたりの画像数
            timeout_seconds: 個別画像処理のタイムアウト
            retry_attempts: 失敗した画像の再試行回数
            memory_limit_mb: ワーカーあたりのメモリ制限（MB）
        """
```

#### ロギング設定

```python
def configure_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    file_path: Optional[str] = None,
) -> None:
    """パイプライン操作のロギングを設定

    Args:
        level: ロギングレベル（"DEBUG", "INFO", "WARNING", "ERROR"）
        format_string: カスタムログフォーマット文字列
        file_path: ログファイルパス（オプション、デフォルトはstdout）
    """
```

### エラータイプ

#### パイプライン例外

```python
class PipelineError(Exception):
    """パイプラインエラーの基底例外"""
    pass

class StepExecutionError(PipelineError):
    """ステップ実行中のエラー"""

    def __init__(self, step_name: str, message: str):
        self.step_name = step_name
        super().__init__(f"ステップ '{step_name}' でのエラー: {message}")

class ImageLoadError(PipelineError):
    """画像ファイル読み込みエラー"""
    pass

class ImageSaveError(PipelineError):
    """画像ファイル保存エラー"""
    pass

class ValidationError(PipelineError):
    """入力検証中のエラー"""
    pass
```

### パフォーマンス監視

#### パフォーマンス指標

```python
@dataclass
class PerformanceMetrics:
    """パイプライン実行のパフォーマンス指標"""

    total_images: int
    processed_images: int
    failed_images: int
    total_time: float
    average_time_per_image: float
    memory_usage_mb: float
    cpu_usage_percent: float

    @classmethod
    def from_pipeline_result(cls, result: PipelineResult) -> "PerformanceMetrics":
        """パイプライン結果から指標を作成"""
        return cls(
            total_images=result.processed_count + result.failed_count,
            processed_images=result.processed_count,
            failed_images=result.failed_count,
            total_time=result.duration_seconds,
            average_time_per_image=result.duration_seconds / max(1, result.processed_count),
            memory_usage_mb=0.0,  # 実行中に設定される
            cpu_usage_percent=0.0,  # 実行中に設定される
        )
```

#### メモリプロファイリング

```python
def profile_memory_usage(func: Callable) -> Callable:
    """関数のメモリ使用量をプロファイルするデコレータ

    Args:
        func: プロファイルする関数

    Returns:
        メモリ使用量を報告するラップされた関数
    """

def get_memory_usage() -> Dict[str, float]:
    """現在のメモリ使用量統計を取得

    Returns:
        メモリ使用量情報を含む辞書
    """
```

### バッチ処理

#### バッチ操作

```python
def process_in_batches(
    pipeline: Pipeline,
    input_paths: List[str],
    batch_size: int = 100,
) -> List[PipelineResult]:
    """メモリ使用量を管理するためにバッチで画像を処理

    Args:
        pipeline: 実行するパイプライン
        input_paths: 入力ファイルパスのリスト
        batch_size: バッチあたりの画像数

    Returns:
        各バッチの結果リスト
    """

def merge_batch_results(results: List[PipelineResult]) -> PipelineResult:
    """複数のバッチ結果を単一結果にマージ

    Args:
        results: バッチ結果のリスト

    Returns:
        結合されたパイプライン結果
    """
```

### 統合例

#### NumPyとの統合

```python
import numpy as np
import flowimds as fi

# 合成画像を作成
images = [
    np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    for _ in range(100)
]

# パイプラインで処理
pipeline = fi.Pipeline(steps=[
    fi.ResizeStep((256, 256)),
    fi.GrayscaleStep(),
])

result = pipeline.run_on_arrays(images)
print(f"{result.processed_count} 枚の合成画像を処理")
```

#### OpenCVとの統合

```python
import cv2
import flowimds as fi

# OpenCVを使用するカスタムステップ
class OpenCVCustomStep:
    def apply(self, image):
        # OpenCV関数を直接使用
        return cv2.medianBlur(image, 5)

# flowimdsと統合
pipeline = fi.Pipeline(steps=[
    fi.ResizeStep((512, 512)),
    OpenCVCustomStep(),
    fi.BinarizeStep(mode="otsu"),
])
```

#### Pillowとの統合

```python
from PIL import Image
import flowimds as fi
import numpy as np

class PillowStep:
    def apply(self, image):
        # PIL Imageに変換
        pil_image = Image.fromarray(image)

        # PIL操作を適用
        pil_image = pil_image.convert('L')  # グレースケール

        # numpyに戻す
        return np.array(pil_image)

pipeline = fi.Pipeline(steps=[
    PillowStep(),
    fi.ResizeStep((256, 256)),
])
```
- 近い将来、よく使う操作を CLI から実行できるようにする計画があります。リポジトリをウォッチしてアップデートを追ってください。
