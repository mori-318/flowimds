# flowimds Usage Guide ドキュメント作成計画

## 調査結果サマリー

### 現状分析

#### ✅ 既に充実している内容
- **概要セクション**: PipelineResultの全フィールド説明
- **実行モード**: 3種類の実行方法 (run, run_on_paths, run_on_arrays) を網羅
- **設定リファレンス**: 主要な設定項目の表形式まとめ
- **ステップリファレンス**: 全6ステップの詳細説明
  - ResizeStep
  - GrayscaleStep
  - BinarizeStep
  - DenoiseStep
  - RotateStep
  - FlipStep
- **サンプルデータセクション**: 実行可能なスクリプト案内

#### ⚠️ 改善・追加が必要な内容

1. **パフォーマンス最適化**
   - `worker_count` パラメータの詳細（現在は設定表に簡潔に記載されているのみ）
   - `log` パラメータによる進捗表示の制御
   - 並列処理のベストプラクティス
   - メモリ使用量に関する考慮事項

2. **エラーハンドリング**
   - 一般的なエラーケースと対処法
   - `failed_files` の活用方法
   - バリデーションエラーの理解

3. **トラブルシューティング**
   - 画像が処理されない場合の診断手順
   - サポートされている画像フォーマットの詳細
   - パス指定の落とし穴

4. **実践的なユースケース / レシピ集**
   - 複数のパイプラインを組み合わせる
   - 条件付き処理の実装
   - カスタムステップの高度な例

5. **制限事項と注意点**
   - 対応画像フォーマット
   - ファイル名の重複処理（flatten時）
   - パフォーマンス上の制約

6. **API詳細**
   - `OutputMapping` オブジェクトの詳細
   - `PipelineSettings` の活用
   - カスタムステップの作成ガイドライン

### 類似ライブラリの構成パターン（参考）

**Pillow (PIL)** のチュートリアル構成:
- 基本クラスの紹介
- 画像の読み書き
- 切り取り・貼り付け・結合
- 幾何学的変換
- 色変換
- 画像強調
- シーケンス処理
- より高度な読み込み
- デコーダー制御

**ベストプラクティス**:
- 段階的な説明（基本→応用）
- 豊富なコード例
- 各セクションで具体的なユースケースを提示
- 注意事項やTipsの明示

---

## ドキュメント更新計画

### Phase 1: 既存コンテンツの強化（優先度: 高）

#### 1.1 パフォーマンスセクションの追加
**位置**: 「Configuring the Pipeline」セクションの後

**内容**:
```markdown
## Performance Tuning

### Parallel Processing

By default, flowimds uses approximately 70% of available CPU cores:

\`\`\`python
# Explicit worker control
pipeline = fi.Pipeline(
    steps=[...],
    input_path="input",
    output_path="output",
    worker_count=8,  # Use 8 worker threads
)
\`\`\`

- `worker_count=None` (default): Auto-detect ~70% of CPU cores
- `worker_count=1`: Sequential processing (useful for debugging)
- `worker_count=0`: Uses all available CPU cores

**Guidelines**:
- For I/O-bound workloads: consider `worker_count = cpu_count * 1.5`
- For CPU-bound workloads: use `worker_count = cpu_count * 0.7`
- Monitor memory usage with large worker counts

### Progress Monitoring

Enable logging to track pipeline execution:

\`\`\`python
pipeline = fi.Pipeline(
    steps=[...],
    log=True,  # Enable progress bar and logs
)
\`\`\`

With `log=True`, you'll see:
- Progress bar (via tqdm)
- Worker/core count information
- Periodic progress updates

### Memory Considerations

- Large images consume more memory during parallel processing
- Reduce `worker_count` if memory becomes constrained
- Consider processing images in batches for very large collections
\`\`\`

**チェック項目**:
- [ ] worker_count の各設定値の挙動を説明
- [ ] log パラメータの効果を明示
- [ ] メモリ使用量のガイドラインを追加
- [ ] 実行例を追加

#### 1.2 エラーハンドリングセクションの追加
**位置**: 「Inspecting PipelineResult」セクションの後

**内容**:
```markdown
## Error Handling

### Understanding Failed Images

When images fail to process, they're recorded in `result.failed_files`:

\`\`\`python
result = pipeline.run()

if result.failed_count > 0:
    print(f"⚠️  {result.failed_count} images failed:")
    for path in result.failed_files:
        print(f"  - {path}")
\`\`\`

**Common failure causes**:
- Unsupported image format
- Corrupted image file
- Insufficient memory
- Permission errors
- Invalid image dimensions

### Handling Validation Errors

Steps validate their parameters during initialization:

\`\`\`python
try:
    step = fi.ResizeStep((0, 100))  # Invalid: width must be positive
except ValueError as e:
    print(f"Configuration error: {e}")
\`\`\`

**Common validation errors**:
- ResizeStep: dimensions must be positive integers
- BinarizeStep: threshold required for mode='fixed'
- DenoiseStep: kernel_size must be odd and ≥ 3
- FlipStep: at least one of horizontal/vertical must be True

### Robust Pipeline Design

\`\`\`python
def process_with_fallback(input_path, output_path):
    pipeline = fi.Pipeline(
        steps=[...],
        input_path=input_path,
        output_path=output_path,
        log=True,
    )
    
    result = pipeline.run()
    
    # Retry failed images with simpler pipeline
    if result.failed_files:
        print(f"Retrying {len(result.failed_files)} failed images...")
        simple_pipeline = fi.Pipeline(
            steps=[fi.ResizeStep((256, 256))],
            output_path=output_path,
        )
        retry_result = simple_pipeline.run_on_paths(result.failed_files)
        print(f"Recovered: {retry_result.processed_count}")
    
    return result
\`\`\`
\`\`\`

**チェック項目**:
- [ ] failed_files の活用方法を例示
- [ ] 一般的なエラーケースをリストアップ
- [ ] バリデーションエラーの例を追加
- [ ] リトライロジックの例を追加

#### 1.3 トラブルシューティングセクションの追加
**位置**: ドキュメント末尾（「Tips and Next Steps」の前）

**内容**:
```markdown
## Troubleshooting

### Images Not Being Processed

**Problem**: `run()` returns 0 processed images

**Solutions**:
1. Verify the input directory exists and contains images
2. Check supported formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif`
3. Enable logging to see discovery details:
   \`\`\`python
   pipeline = fi.Pipeline(..., log=True)
   result = pipeline.run()
   \`\`\`
4. Try `recursive=True` if images are in subdirectories

### Output Files Not Created

**Problem**: Pipeline completes but no output files appear

**Solutions**:
1. Ensure `output_path` directory exists (created automatically)
2. Check write permissions for output directory
3. Inspect `result.failed_files` for specific failures
4. Verify output mappings:
   \`\`\`python
   for mapping in result.output_mappings:
       print(f"{mapping.input_path} -> {mapping.output_path}")
   \`\`\`

### File Name Collisions (Flattened Output)

**Problem**: Files overwritten when `preserve_structure=False`

**Behavior**: flowimds automatically appends `_no{N}` suffixes to duplicates:
- `image.png` → `image.png`
- `image.png` (duplicate) → `image_no2.png`
- `image.png` (duplicate) → `image_no3.png`

### Performance Issues

**Problem**: Processing is slower than expected

**Solutions**:
1. Increase `worker_count` for I/O-bound tasks
2. Reduce `worker_count` if memory-constrained
3. Profile individual steps to identify bottlenecks
4. Consider pre-filtering inputs with `run_on_paths`

### Japanese File Names

**Note**: flowimds uses OpenCV's special handling for non-ASCII paths. Japanese characters in file names and paths are fully supported.
\`\`\`

**チェック項目**:
- [ ] よくある問題とソリューションを5-7個リストアップ
- [ ] ファイル名重複時の挙動を明示
- [ ] 日本語ファイル名対応を明記
- [ ] デバッグのヒントを追加

### Phase 2: 高度な内容の追加（優先度: 中）

#### 2.1 実践的なレシピ集
**位置**: 新規セクション「Common Recipes」として「Built-in Step Reference」の後

**内容例**:
- データ拡張パイプライン（回転+フリップ）
- 複数出力フォーマットへの変換
- カスタムフィルタの統合
- 条件付き処理（ファイル名パターンマッチング）

**チェック項目**:
- [ ] 5-7個の実用的なレシピを作成
- [ ] 各レシピに説明と完全なコード例を添付
- [ ] 実行可能性を検証

#### 2.2 カスタムステップガイドの拡張
**位置**: 「Built-in Step Reference」の後

**内容**:
- PipelineStepプロトコルの詳細
- クラスベースのカスタムステップ
- 関数ベースのステップ
- ステップの状態管理
- パラメータバリデーション例

**チェック項目**:
- [ ] Protocol の仕様を明示
- [ ] 複数のカスタムステップ実装例を追加
- [ ] ベストプラクティスをリストアップ

### Phase 3: APIリファレンスの充実（優先度: 中）

#### 3.1 OutputMapping詳細
**内容**:
- フィールド説明
- 活用例（マッピングの保存、統計情報の抽出）

#### 3.2 PipelineSettings詳細
**内容**:
- 全フィールドの説明
- ログ・監査での活用例

**チェック項目**:
- [ ] 各データクラスのフィールドを文書化
- [ ] 実用例を2-3個追加

### Phase 4: メンテナンスと品質管理（優先度: 低）

#### 4.1 制限事項セクション
**位置**: ドキュメント末尾

**内容**:
- サポート画像フォーマット一覧
- パフォーマンス制約
- 既知の問題

#### 4.2 バージョン互換性
**内容**:
- 主要バージョン間の変更点
- 非推奨API

---

## ドキュメント作成チェックリスト

### ✅ コンテンツ完全性

#### 必須項目
- [x] 概要と目的の明示
- [x] 3つの実行モード（run, run_on_paths, run_on_arrays）の説明
- [x] PipelineResultの全フィールド説明
- [x] 全ビルトインステップのリファレンス
- [x] 設定パラメータの一覧
- [ ] **パフォーマンスチューニングガイド**
- [ ] **エラーハンドリング詳細**
- [ ] **トラブルシューティングセクション**

#### 推奨項目
- [ ] 実践的なユースケース集（5個以上）
- [ ] カスタムステップの詳細ガイド
- [ ] OutputMapping / PipelineSettings の活用例
- [ ] 制限事項の明示
- [ ] よくある質問 (FAQ)

### ✅ コード例の品質

#### 基準
- [x] すべてのコード例が実行可能
- [x] import文を含む完全な例
- [x] コメントで各ステップを説明
- [ ] **エラーハンドリング例を含む**
- [ ] **複数の難易度レベル（初級・中級・上級）**

#### 検証項目
- [ ] 各コード例を実際に実行してテスト
- [ ] 出力例を明示（可能な場合）
- [ ] エッジケースへの対応を含む

### ✅ 文章の明瞭性

#### 構成
- [x] セクションが論理的に配置されている
- [x] 見出し階層が適切
- [ ] **目次 (Table of Contents) の追加**
- [ ] **各セクションに導入文**

#### スタイル
- [x] 一貫した用語の使用
- [x] 簡潔で明確な文章
- [ ] **技術用語の定義または初出時の説明**
- [ ] **Note/Warning/Tip ボックスの活用**

### ✅ アクセシビリティ

#### 対象読者
- [x] 初心者向けの説明が含まれている
- [x] 上級者向けの詳細も提供
- [ ] **前提知識の明示**
- [ ] **関連ドキュメントへのリンク**

#### ナビゲーション
- [ ] **内部リンク（セクション間参照）の追加**
- [x] 外部リソースへのリンク（該当する場合）
- [ ] **「次のステップ」ガイドの明示**

### ✅ 正確性

#### コードとの整合性
- [x] 全APIが最新のコードと一致
- [x] デフォルト値が正確
- [x] パラメータ型が正確
- [ ] **バージョン番号の明記**

#### 検証プロセス
- [ ] コードレビューの実施
- [ ] 技術的な正確性の確認
- [ ] ユーザーテスト（可能な場合）

### ✅ メンテナンス性

#### 更新容易性
- [x] セクションが独立している
- [ ] **コード例が別ファイルで管理**（オプション）
- [ ] **変更履歴の記録**

#### 一貫性
- [x] 用語集の維持
- [x] スタイルガイドの遵守
- [ ] **テンプレートの使用**（新規セクション追加時）

---

## 注意点とベストプラクティス

### 📝 文章作成時の注意点

1. **コード例の正確性**
   - すべてのコード例を実際に実行して動作確認
   - import文を省略しない
   - エラーケースも含める

2. **バージョン依存の記載**
   - 特定のバージョンに依存する機能は明記
   - 非推奨機能には警告を表示

3. **用語の一貫性**
   - "step" vs "transform" → 「step」に統一
   - "image" vs "picture" → 「image」に統一
   - "directory" vs "folder" → 「directory」に統一

4. **例の多様性**
   - 単純な例から複雑な例へ段階的に
   - 実世界のユースケースを反映
   - エッジケースも含める

### 🔍 レビュー時のチェックポイント

1. **技術的正確性**
   - API仕様との整合性
   - パラメータ型とデフォルト値の確認
   - コード例の実行可能性

2. **可読性**
   - セクションの論理的な流れ
   - 適切な見出しレベル
   - コードコメントの充実度

3. **完全性**
   - すべての主要機能がカバーされているか
   - エッジケースへの言及
   - 既知の制限事項の記載

4. **ユーザビリティ**
   - 初心者にも理解可能か
   - 検索しやすい構成か
   - 実践的な例が含まれているか

### 🌐 多言語対応の注意点

1. **日本語版との同期**
   - 英語版と日本語版の内容を一致させる
   - 翻訳時のニュアンス調整
   - 文化的な違いへの配慮

2. **コード例のローカライゼーション**
   - ファイルパスの例（Unix vs Windows）
   - 文字エンコーディング関連の注意事項

### 📊 メトリクス（更新後に測定）

- [ ] ドキュメント全体の単語数
- [ ] コード例の数
- [ ] セクション数
- [ ] 内部リンク数
- [ ] 外部参照数

---

## 実装スケジュール案

### Week 1: Phase 1（優先度: 高）
- Day 1-2: パフォーマンスセクション作成
- Day 3-4: エラーハンドリングセクション作成
- Day 5: トラブルシューティングセクション作成
- Day 6-7: レビューと修正

### Week 2: Phase 2（優先度: 中）
- Day 1-3: レシピ集作成
- Day 4-5: カスタムステップガイド拡張
- Day 6-7: レビューと修正

### Week 3: Phase 3-4 + 品質管理
- Day 1-2: APIリファレンス充実
- Day 3: 制限事項セクション
- Day 4-5: 全体レビュー
- Day 6-7: ユーザーテストとフィードバック反映

---

## 参考リソース

### 類似プロジェクトのドキュメント
- [Pillow Documentation](https://pillow.readthedocs.io/)
- [scikit-image User Guide](https://scikit-image.org/docs/stable/user_guide.html)
- [OpenCV-Python Tutorials](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)

### ドキュメント作成ガイド
- [Write the Docs](https://www.writethedocs.org/)
- [Google Developer Documentation Style Guide](https://developers.google.com/style)

### Pythonドキュメント規約
- [PEP 257 - Docstring Conventions](https://www.python.org/dev/peps/pep-0257/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

---

## まとめ

現在の `docs/usage.md` は基本的な機能を網羅した良質なドキュメントですが、以下の点を強化することでさらに実用的になります：

### 🎯 最優先事項
1. **パフォーマンスチューニング** - worker_count と log の詳細
2. **エラーハンドリング** - failed_files の活用方法
3. **トラブルシューティング** - よくある問題と解決策

### 🔧 次の段階
4. **実践的なレシピ集** - 複数のユースケース例
5. **カスタムステップガイド** - Protocol の詳細説明
6. **APIリファレンス** - OutputMapping と PipelineSettings

この計画に従ってドキュメントを更新することで、初心者から上級者まで幅広いユーザーのニーズに応えられる包括的なガイドが完成します。
