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
)

result = pipeline.run(input_path="/path/to/input", recursive=True)  # str でも pathlib.Path でも可
result.save("/path/to/output", preserve_structure=True)
print(f"Processed {result.processed_count} images in {result.duration_seconds:.2f}s")
```

この形式は、ライブラリにディレクトリ走査を任せたいときに便利です。サポートされる拡張子（`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif`）を自動で収集します。`run(recursive=True)` でサブディレクトリをたどり、`save(preserve_structure=True)` で入力の階層構造を出力側に再現できます。

### ファイルリストに対して実行する

```python
import flowimds as fi

paths = [
    "samples/input/receipt.png",
    "samples/input/avatar.jpg",
]

pipeline = fi.Pipeline(
    steps=[fi.ResizeStep((256, 256)), fi.BinarizeStep(mode="otsu")],
)

result = pipeline.run(input_paths=paths)
result.save("samples/output")
for mapping in result.output_mappings:
    print(f"{mapping.input_path} -> {mapping.output_path}")
```

`input_paths` は対象ファイルを事前に把握している場合や、複数のディレクトリにまたがる入力をまとめて処理したい場合に有効です。`run()` を呼ぶたびに `input_path` / `input_paths` のどちらかを指定でき、引数で渡した方が優先されます。

### `input_arrays` でメモリ内だけで完結させる

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

result = pipeline.run(input_arrays=images)
print(f"Got {len(result.processed_images)} transformed images")
```

`input_arrays` はファイルシステムを使わずに NumPy 配列だけを処理します。入力イテラブルの各要素が NumPy 配列かどうかを検証し、変換結果は `PipelineResult.processed_images` に蓄えられます。必要に応じて `result.save(...)` で任意のディレクトリへ書き出してください。

`input_arrays` だけを利用する場合、`Pipeline` の初期化時にパスを設定する必要はありません。後から `run(input_path=...)` や `run(input_paths=...)` を呼ぶときは、必要な入力をその都度 `run(...)` に渡してください。ファイルへ保存したい場合は `result.save(output_path)` を呼びます。

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

import numpy as np
result = pipeline.run(input_arrays=[np.zeros((1, 1, 3), dtype=np.uint8)])
summarise(result)
```

`settings` には実行時設定（入力・出力ディレクトリ、再帰フラグなど）が含まれるので、ログや監査用途に便利です。

## エラーハンドリング

### 失敗した画像の理解

画像の処理が失敗した場合、`result.failed_files` に詳細なエラー情報と共に記録されます：

```python
result = pipeline.run(input_path="/path/to/input")

if result.failed_count > 0:
    print(f"⚠️  {result.failed_count} 枚の画像が失敗しました:")
    for path in result.failed_files:
        print(f"  - {path}")

    # 失敗した画像を別のパイプラインでリトライ
    if result.failed_files:
        print("よりシンプルなパイプラインでリトライ中...")
        retry_pipeline = fi.Pipeline(
            steps=[fi.ResizeStep((256, 256))],  # 最小限の処理
        )
        retry_result = retry_pipeline.run(input_paths=result.failed_files)
        retry_result.save("retries")
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
                log=True,
            )
        elif complexity == "simple":
            return fi.Pipeline(
                steps=[fi.ResizeStep((256, 256)), fi.GrayscaleStep()],
                log=True,
            )
        else:  # minimal
            return fi.Pipeline(
                steps=[fi.ResizeStep((128, 128))],
                log=True,
            )

    # まず完全なパイプラインを試行
    result = create_pipeline("full").run(input_path=input_path)
    result.save(output_path)

    # 失敗した画像をよりシンプルなパイプラインでリトライ
    initial_failed_count = len(result.failed_files)
    failed_files = result.failed_files.copy()
    retry_attempts = 0

    while failed_files and retry_attempts < max_retries:
        retry_attempts += 1
        complexity = ["simple", "minimal"][retry_attempts - 1]

        print(f"リトライ {retry_attempts}/{max_retries}: {complexity} パイプライン...")
        retry_result = create_pipeline(complexity).run(input_paths=failed_files)
        retry_result.save(output_path)

        # 失敗ファイルリストを更新
        recovered = {str(mapping.input_path) for mapping in retry_result.output_mappings}
        newly_failed = set(failed_files) - recovered
        failed_files = list(newly_failed)

        print(f"  回復: {retry_result.processed_count}, 依然として失敗: {len(failed_files)}")

    # 最終サマリー
    print(f"\n最終結果:")
    print(f"  正常に処理: {result.processed_count}")
    print(f"  リトライで回復: {initial_failed_count - len(failed_files)}")
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
    worker_count=8,  # 8つのワーカースレッドを使用
)

result = pipeline.run(input_path="input")
result.save("output")
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
    log=True,  # プログレスバーとログを有効化
)

result = pipeline.run(input_path="large_dataset")
result.save("processed")
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
            worker_count=4,  # メモリに保守的な設定
        )
        result = pipeline.run(input_paths=batch)
        result.save(output_dir)
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
)

result = pipeline.run(input_path="input")
result.save("output")
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

result = pipeline.run(input_arrays=[np.zeros((1, 1, 3), dtype=np.uint8)])
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
    log=True,
    worker_count=4,
)

# パイプラインを実行
result = custom_pipeline.run(input_path="input")
result.save("output")
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
        recursive: bool = False,
        preserve_structure: bool = False,
        worker_count: Optional[int] = None,
        log: bool = False,
    ):
        """新しいパイプラインを初期化

        Args:
            steps: PipelineStepプロトコルを実装する処理ステップのリスト
            recursive: サブディレクトリも走査するか（デフォルト: False）
            preserve_structure: ディレクトリ構造を維持するか（デフォルト: False）
            worker_count: 並列ワーカー数（デフォルト: None = CPUコアの約70%）
            log: 詳細なロギングを有効にする（デフォルト: False）
        """
```

**メソッド:**

- `run(input_path=..., input_paths=..., input_arrays=...) -> PipelineResult`: 入力ソースを指定して実行
- `PipelineResult.save(output_dir) -> None`: 変換済み画像を任意のディレクトリへ保存

**例:**
```python
pipeline = fi.Pipeline(
    steps=[fi.ResizeStep((512, 512)), fi.GrayscaleStep()],
    worker_count=8,
    log=True,
)

# ディレクトリで実行
result = pipeline.run(input_path="input")
result.save("output")

# 特定のファイルで実行
from pathlib import Path
specific_files = [Path("img1.jpg"), Path("img2.jpg")]
result = pipeline.run(input_paths=specific_files)
result.save("output")

# numpy配列で実行
import numpy as np
arrays = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)]
result = pipeline.run(input_arrays=arrays)
result.save("output")
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
)

result = pipeline.run(input_path="input")
result.save("output")
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

result = pipeline.run(input_arrays=images)
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
