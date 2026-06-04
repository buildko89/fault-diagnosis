# Analog Circuit Fault Diagnosis Tool

アナログ回路におけるハードウェア故障（枝アドミタンスの変動）を検知・診断するためのPythonベースの診断ツール（PoC）です。
Huang-Lin-Liu (1983) および Togawa-Matsumoto (1984) の理論に基づき、大規模回路にも対応可能な最新のスパースモデリング（OMP）を導入しています。

## 💡 このツールの「何がうれしいのか？」「何がすごいのか？」

### 1. 「分解せずに」内部の故障箇所がわかる（非破壊検査）
通常、電子回路の故障箇所を特定するには、回路基板上のあらゆる部品に直接テスター（プローブ）を当てる必要があります。しかし、本ツールを使えば、**あらかじめ設定された少数の外部端子（ADC計測ノード）の電圧を測るだけ**で、回路内部の「どの部品が」「どの程度」劣化・故障しているのかを数学的に推定できます。

### 2. 「組み合わせ爆発」を抑え、大規模回路でも一瞬で診断（OMPの威力）
「ノードが1000個ある回路から、故障している3個を見つける」場合、総当たり（全探索）では約1億6000万通りの計算が必要になり、これまでは実用的な時間で終わりませんでした。本ツールは **「スパースモデリング（Orthogonal Matching Pursuit: OMP）」** という圧縮センシング技術を採用することでこの天文学的な計算をスキップし、**大規模な回路でも一瞬（ミリ秒〜秒単位）で故障箇所を特定**できます。

### 3. メモリを極限まで節約する「スパース行列」アーキテクチャ
大規模な回路の計算を愚直に行うと、巨大な行列メモリが必要になりPCがフリーズしてしまいます。本ツールでは計算のコア部分をすべて `scipy.sparse` (疎行列) に書き換えており、**実質的に部品が繋がっている部分のメモリしか消費しません**。これにより、一般的なPCでも数千〜数万ノード規模の回路方程式の構築と求解が可能になっています。

## 🌟 主な機能

* **k-Node Testability の判定**: 回路のトポロジーとアクセス可能ノード（測定端子）の配置から、最大 $k$ 個の故障を特定可能かを事前判定します。
* **大規模回路対応（Sparse Matrix）**: `scipy.sparse` を用いた疎行列計算により、数千ノード規模の回路方程式も高速かつ省メモリに構築・求解します。
* **スパースモデリングによる高速診断（OMP）**: 従来の組み合わせ爆発を引き起こす全探索（Exhaustive Search）に代わり、`scikit-learn` の **Orthogonal Matching Pursuit (OMP)** を用いて故障ノードを高速に推定します。
* **自動レポート生成**: 診断結果から自動的にMarkdown形式のレポートと、`matplotlib` / `networkx` を用いたトポロジー可視化画像を出力します。

---

## 🛠 システム要件 (Requirements)

* Python 3.9+
* `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `networkx`, `pyyaml`
* `pytest` (テスト実行用)

### インストール

仮想環境の利用を推奨します。

```bash
# 依存パッケージのみ
pip install -r requirements.txt

# もしくはパッケージとして（CLI コマンド analog-fault を含む / 開発用依存込み）
pip install -e ".[dev]"
```

---

## 📂 プロジェクト構成

```
.
├── analog_fault/        # 本体パッケージ
│   ├── schema.py        # 回路定義 (dataclass) と YAML ロード・バリデーション
│   ├── circuit.py       # 疎行列によるアドミタンス行列 (A, Yb, Y) 構築
│   ├── testability.py   # k-node テスタビリティ判定 (最大流 / 頂点独立パス)
│   ├── simulate.py      # ΔV と転送インピーダンス行列 Z_mn の算出
│   ├── diagnose.py      # 故障ノード診断 (auto / exhaustive / S-OMP) と枝再構築
│   ├── evaluate.py      # モンテカルロ精度評価 (公差・ノイズ)
│   ├── reporter.py      # Markdown レポート + トポロジー/ΔV 図の生成
│   └── cli.py           # CLI (testability / diagnose / evaluate)
├── examples/            # サンプル回路 (bridge.yaml, ladder.yaml)
├── tests/               # pytest 回帰テスト
├── prototype.py         # デモスクリプト
├── requirements.txt
└── pyproject.toml
```

---

## 🏗 アーキテクチャと実装内容

本パッケージ（`analog_fault`）は、以下の主要モジュールから構成されています。

| モジュール名 | 主な役割 | 実装技術・ハイライト |
|:---|:---|:---|
| `circuit.py` | 回路網のモデリング | `sp.lil_matrix` と `csr_matrix` を用いた疎行列ベースのアドミタンス行列 ($Y$, $A$, $Y_b$) 構築 |
| `testability.py` | 故障診断可能性の事前判定 | NetworkXを用いたノード間の独立パス（Vertex Disjoint Paths）探索 |
| `simulate.py` | 回路シミュレーション | `scipy.sparse.linalg.spsolve` を利用した安定した連立方程式求解（$\Delta V$ の算出） |
| `diagnose.py` | 故障ノード・アドミタンス変動量の推定 | Strategyパターン（`method='auto'` / `'exhaustive'` / `'omp'`(S-OMP)）。枝アドミタンス再構築では L2正則化（Ridge回帰）も選択可 |
| `reporter.py` | 解析結果の可視化とレポート生成 | ヘッドレスモード(`Agg`)でのトポロジー図・電圧偏差グラフ出力および Markdown レポート出力 |

### 故障診断アルゴリズムの比較

| 診断メソッド | 呼び出し指定 | 特徴 | 適用シーン |
|:---|:---|:---|:---|
| **自動選択** | `method='auto'`<br>*(デフォルト)* | 組合せ数が閾値以下なら厳密な全探索、それを超える大規模回路でのみ OMP に自動フォールバック。小規模では精度、大規模では速度を両立。 | 通常はこれを使う。 |
| **OMP (S-OMP)** | `method='omp'` | 複数励起に対応した同時直交マッチング追跡（貪欲法）。計算量は小さいが、**対称性の高い回路や内部（非アクセス）ノードの故障では取りこぼす場合がある近似解法**。 | 全探索が現実的でない大規模回路。 |
| **全探索** | `method='exhaustive'` | 故障ノードの組み合わせを総当たりし、最小二乗法で厳密解を探索。計算量は $O(_nC_k)$。対称回路でも確実。 | 小〜中規模回路、理論検証、テストでの完全一致確認。 |

> ⚠️ **精度に関する注意**: OMP / L1 などのスパース近似は、ブリッジのような対称回路や内部ノードの故障（例: 故障signatureが特定の組み合わせでしか現れないケース）を構造的に取りこぼすことがあります。小規模回路では既定の `auto`（=全探索）を使うことを推奨します。

---

## 🚀 使い方 (Usage)

### 1. デモの実行

同梱されているデモスクリプト `prototype.py` を実行すると、LadderネットワークとBridgeネットワークの2種類の回路に対するシミュレーションと故障診断が行われます。

```bash
python prototype.py
```

### 2. コードからの呼び出し例

```python
import numpy as np
from analog_fault.schema import CircuitConfig, Element
from analog_fault.circuit import AnalogCircuit
from analog_fault.simulate import calculate_delta_v
from analog_fault.diagnose import diagnose_node_faults

# 1. 回路の定義
config = CircuitConfig(
    name="sample",
    reference=0,
    nodes=[0, 1, 2, 3],
    accessible=[1, 2],
    elements=[
        Element("R1", "R", 1, 0, 1.0),
        Element("R2", "R", 1, 2, 1.0),
        Element("R3", "R", 2, 3, 1.0),
    ]
)
circuit = AnalogCircuit(config)

# 2. 観測データの取得 (シミュレーション)
excitations = [np.array([1.0, 0.0, 0.0])]
delta_v_ms, Z_mn = calculate_delta_v(circuit, excitations)

# 3. 故障診断の実行 (method 既定の 'auto': 小規模は厳密な全探索を自動選択)
result = diagnose_node_faults(circuit, Z_mn, delta_v_ms, max_faults=1)
print("故障ノード:", result['best']['support'])
```

---

## 📊 レポート自動生成機能

診断スクリプト（`prototype.py` 等）の実行後、自動的に `./report` ディレクトリが作成され、以下のファイルが出力されます。

1. **`diagnosis_report.md`**:
   * OMPと全探索（Exhaustive）の実行時間（Performance Comparison）の比較表
   * 特定された故障ノードのサマリー
2. **`topology.png`**:
   * 回路の結線状態を示すトポロジーグラフ。
   * <span style="color:red">**赤色**</span>: 故障と診断されたノード
   * <span style="color:skyblue">**青色**</span>: アクセス可能（ADC計測）ノード
   * <span style="color:gray">**灰色**</span>: アクセス不可（内部）ノード
3. **`delta_v.png`**:
   * アクセス可能ノードにおける、観測された電圧偏差 $\Delta V_m$ を示す棒グラフ。

---

## ✅ テストの実行

開発やリファクタリング後の完全性を担保するため、`pytest` による回帰テストを完備しています。

```bash
pytest tests/
```
*(回路の回帰テストは厳密なアサーションのため `method='exhaustive'` を、補助モジュールのテストは schema 検証 / YAML ロード / テスタビリティ偽判定 / 評価 / Ridge 再構築をカバーします)*

---

## ⚠️ 制約事項 (Limitations)

* **素子は実コンダクタンス (`type: R`) のみ対応**。容量・インダクタンス（複素アドミタンス）や AC・周波数掃引には未対応です（PoC のスコープ）。
* **OMP は近似解法**です。対称性の高い回路や内部（非アクセス）ノードの故障では取りこぼす場合があるため、小〜中規模では既定の `auto`（自動で全探索）を使用してください。詳細は上記「故障診断アルゴリズムの比較」を参照。
* 診断段の転送インピーダンス行列 `Z_mn` は密行列として保持されるため、超大規模回路ではメモリ・計算量に注意が必要です。

詳しい解析と改善履歴は [`Docs/Report20260604/改善レポート.md`](Docs/Report20260604/改善レポート.md) を参照してください。

---

## 📄 ライセンス (License)

本リポジトリは [MIT License](LICENSE) の下で公開されています。
