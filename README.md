# Circuit Fault Diagnosis Tool

電子回路におけるハードウェア故障（枝アドミタンスの変動）を検知・診断するためのPythonベースの診断ツール（PoC）です。
抵抗(R)・容量(C)・インダクタンス(L)を含む回路を、複数周波数(AC)で診断・故障素子の種別（R/C/L）まで同定できます。
Huang-Lin-Liu (1983) および Togawa-Matsumoto (1984) の理論に基づき、大規模回路にも対応可能な最新のスパースモデリング（OMP）を導入しています。

## 💡 このツールの「何がうれしいのか？」「何がすごいのか？」

### 1. 「分解せずに」内部の故障箇所がわかる（非破壊検査）
通常、電子回路の故障箇所を特定するには、回路基板上のあらゆる部品に直接テスター（プローブ）を当てる必要があります。しかし、本ツールを使えば、**あらかじめ設定された少数の外部端子（ADC計測ノード）の電圧を測るだけ**で、回路内部の「どの部品が」「どの程度」劣化・故障しているのかを数学的に推定できます。

### 2. 「組み合わせ爆発」を抑え、大規模回路でも一瞬で診断（OMPの威力）
「ノードが1000個ある回路から、故障している3個を見つける」場合、総当たり（全探索）では約1億6000万通りの計算が必要になり、これまでは実用的な時間で終わりませんでした。本ツールは **「スパースモデリング（Orthogonal Matching Pursuit: OMP）」** という圧縮センシング技術を採用することでこの天文学的な計算をスキップし、**大規模な回路でも一瞬（ミリ秒〜秒単位）で故障箇所を特定**できます。

### 3. メモリを極限まで節約する「スパース行列」アーキテクチャ
大規模な回路の計算を愚直に行うと、巨大な行列メモリが必要になりPCがフリーズしてしまいます。本ツールでは計算のコア部分をすべて `scipy.sparse` (疎行列) に書き換えており、**実質的に部品が繋がっている部分のメモリしか消費しません**。これにより、一般的なPCでも数千〜数万ノード規模の回路方程式の構築と求解が可能になっています。

## 🌟 主な機能

* **R/C/L・AC（複数周波数）対応**: 抵抗だけでなく容量・インダクタンスを含む回路を、複数の測定周波数で診断できます。複素アドミタンスで解き、故障枝の周波数依存性から **R/C/L の種別と変動量**を同定します。
* **k-Node Testability の判定**: 回路のトポロジーとアクセス可能ノード（測定端子）の配置から、最大 $k$ 個の故障を特定可能かを事前判定します。
* **大規模回路対応（Sparse Matrix）**: `scipy.sparse` を用いた疎行列計算により、数千ノード規模の回路方程式も高速かつ省メモリに構築・求解します。
* **スパースモデリングによる高速診断（OMP）**: 従来の組み合わせ爆発を引き起こす全探索（Exhaustive Search）に代わり、`scikit-learn` の **Orthogonal Matching Pursuit (OMP)** を用いて故障ノードを高速に推定します。
* **自動レポート生成**: 診断結果から自動的にMarkdown形式のレポートと、`matplotlib` / `networkx` を用いたトポロジー可視化画像を出力します。

---

## 🛠 システム要件 (Requirements)

* Python 3.9+
* `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `networkx`, `pyyaml`
* `pytest` (テスト実行用)
* `streamlit` (GUI/Web UI を使う場合のみ)

### インストール

仮想環境の利用を推奨します。

```bash
# 依存パッケージのみ
pip install -r requirements.txt

# もしくはパッケージとして（CLI コマンド fault を含む / 開発用依存込み）
pip install -e ".[dev]"

# GUI（Web UI）も使う場合（Streamlit を追加）
pip install -e ".[gui]"
```

---

## 📂 プロジェクト構成

```
.
├── fault/        # 本体パッケージ
│   ├── schema.py        # 回路定義 (dataclass) と YAML ロード・バリデーション
│   ├── circuit.py       # 疎行列によるアドミタンス行列 (A, Yb, Y) 構築
│   ├── testability.py   # k-node テスタビリティ判定 (最大流 / 頂点独立パス)
│   ├── simulate.py      # 周波数ごとの ΔV と転送インピーダンス Z_mn (MeasurementBlock)
│   ├── diagnose.py      # 故障ノード診断 (auto / exhaustive / S-OMP) と枝再構築・R/C/L 分類
│   ├── evaluate.py      # モンテカルロ精度評価 (公差・ノイズ・複数周波数)
│   ├── reporter.py      # Markdown レポート + トポロジー/ΔV 図の生成（図ビルダは GUI と共用）
│   ├── service.py       # フレームワーク非依存サービス層 (CLI/GUI 共通の load/testability/diagnose/evaluate)
│   └── cli.py           # CLI (testability / diagnose / evaluate)
├── fault_gui/           # Streamlit 製 Web UI (app + Circuit/Testability/Diagnose/Evaluate タブ)
├── examples/            # サンプル回路 (bridge.yaml, ladder.yaml, rc_bridge.yaml)
├── tests/               # pytest 回帰テスト
├── prototype.py         # デモスクリプト
├── requirements.txt
└── pyproject.toml
```

---

## 🏗 アーキテクチャと実装内容

本パッケージ（`fault`）は、以下の主要モジュールから構成されています。

| モジュール名 | 主な役割 | 実装技術・ハイライト |
|:---|:---|:---|
| `circuit.py` | 回路網のモデリング | `sp.lil_matrix` と `csr_matrix` を用いた疎行列ベースのアドミタンス行列 ($Y$, $A$, $Y_b$) 構築 |
| `testability.py` | 故障診断可能性の事前判定 | NetworkXを用いたノード間の独立パス（Vertex Disjoint Paths）探索 |
| `simulate.py` | 回路シミュレーション | 周波数ごとに複素 $Y(\omega)$ を `spsolve` で求解し `MeasurementBlock`（$\Delta V$, $Z_{mn}$）を生成 |
| `diagnose.py` | 故障ノード・アドミタンス変動量の推定 | Strategyパターン（`method='auto'` / `'exhaustive'` / `'omp'`(S-OMP)）。複数周波数を結合して診断し、枝再構築では R/C/L 種別同定や L2正則化（Ridge回帰）も選択可 |
| `reporter.py` | 解析結果の可視化とレポート生成 | ヘッドレスモード(`Agg`)でのトポロジー図・電圧偏差グラフ出力および Markdown レポート出力 |

### 故障診断アルゴリズムの比較

| 診断メソッド | 呼び出し指定 | 特徴 | 適用シーン |
|:---|:---|:---|:---|
| **自動選択** | `method='auto'`<br>*(デフォルト)* | 組合せ数が閾値以下なら厳密な全探索、それを超える大規模回路でのみ OMP に自動フォールバック。小規模では精度、大規模では速度を両立。 | 通常はこれを使う。 |
| **OMP (S-OMP)** | `method='omp'` | 複数励起に対応した同時直交マッチング追跡（貪欲法）。計算量は小さいが、**対称性の高い回路や内部（非アクセス）ノードの故障では取りこぼす場合がある近似解法**。 | 全探索が現実的でない大規模回路。 |
| **全探索** | `method='exhaustive'` | 故障ノードの組み合わせを総当たりし、最小二乗法で厳密解を探索。計算量は $O(_nC_k)$。対称回路でも確実。 | 小〜中規模回路、理論検証、テストでの完全一致確認。 |

> ⚠️ **精度に関する注意**: OMP / L1 などのスパース近似は、ブリッジのような対称回路や内部ノードの故障（例: 故障signatureが特定の組み合わせでしか現れないケース）を構造的に取りこぼすことがあります。小規模回路では既定の `auto`（=全探索）を使うことを推奨します。
>
> 💡 **多周波数の効果**: 複数の周波数で測定すると、support 選択に対する独立な制約が増え、識別可能性が向上します（リアクタンス素子を含む回路で特に有効）。`frequencies` / `--freq` に複数の値を与えてください。

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
from fault.schema import CircuitConfig, Element
from fault.circuit import Circuit
from fault.simulate import calculate_delta_v
from fault.diagnose import diagnose_node_faults

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
circuit = Circuit(config)

# 2. 観測データの取得 (シミュレーション)
excitations = [np.array([1.0, 0.0, 0.0])]
delta_v_ms, Z_mn = calculate_delta_v(circuit, excitations)

# 3. 故障診断の実行 (method 既定の 'auto': 小規模は厳密な全探索を自動選択)
result = diagnose_node_faults(circuit, Z_mn, delta_v_ms, max_faults=1)
print("故障ノード:", result['best']['support'])
```

### 3. AC（複数周波数）での診断と R/C/L 種別同定

容量(C)・インダクタンス(L)を含む回路では、`frequencies`（Hz）を指定し `calculate_measurements` で周波数ごとの観測ブロックを得ます。`diagnose_node_faults` はブロックのリストをそのまま受け取れます。

```python
from fault.simulate import calculate_measurements
from fault.diagnose import diagnose_node_faults, reconstruct_branch_faults

# C/L を含む回路（frequencies は Hz、内部で ω=2πf に変換）
config = CircuitConfig(
    name="rc", reference=0, nodes=[0, 1, 2], accessible=[1, 2],
    frequencies=[1000.0, 5000.0],
    elements=[
        Element("R1", "R", 1, 0, 1.0),
        Element("C1", "C", 1, 2, 1.0e-3),   # value = 容量[F]
    ],
)
circuit = Circuit(config)
excitations = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]

# 周波数ごとの観測ブロックを取得し、そのまま診断
blocks = calculate_measurements(circuit, excitations,
                                faulty_elements={"C1": 0.5e-3}, frequencies=config.frequencies)
result = diagnose_node_faults(circuit, blocks, None, max_faults=2)

# 故障枝の複素アドミタンス偏差と R/C/L 種別を再構築
branch = reconstruct_branch_faults(circuit, sorted(result['best']['support']),
                                   excitations, blocks, result)
# 例: branch['C1']['classification'] == 'C', branch['C1']['delta_C'] ≈ -0.5e-3
```

### 4. CLI からの実行

```bash
# テスタビリティ判定
fault testability examples/bridge.yaml --k 2

# DC 診断
fault diagnose examples/bridge.yaml --fault R4=0.1 --k 2

# AC 診断（--freq か、YAML の frequencies を使用）
fault diagnose examples/rc_bridge.yaml --fault C4=0.5e-3 --k 2 --freq 1000,5000

# モンテカルロ評価
fault evaluate examples/rc_bridge.yaml --fault C4=0.5e-3 --k 2 --trials 100
```

*(`pip install -e .` していない場合は `python -m fault.cli ...` でも実行できます)*

### 5. GUI（Web UI）からの実行

CLI と同じ機能（テスタビリティ判定 / 故障診断 / モンテカルロ評価）を、回路の読込から結果・図の表示まで
ブラウザ上の 1 画面で実行できる **Streamlit 製 Web UI** を同梱しています。コア計算ロジックは CLI と共通の
サービス層（`fault/service.py`）を経由しており、CLI と同じ結果が得られます。

```bash
# 事前に GUI 依存を導入（Streamlit）
pip install -e ".[gui]"

# 起動（いずれでも可）
fault-gui
# もしくは
streamlit run fault_gui/app.py
# もしくは
python -m fault_gui
```

起動するとブラウザが開きます。サイドバーで回路を読み込み（`examples/` から選択 / YAML アップロード /
パス入力）、共通パラメータ（k・周波数）を設定したうえで、4 つのタブを操作します。

| タブ | 内容 |
|:---|:---|
| **Circuit** | トポロジー図・回路サマリ・素子一覧 |
| **Testability** | k-node テスタビリティ判定（Testable バッジ＋内部ノード接続度） |
| **Diagnose** | 故障素子を表で指定 → 診断結果（status / 候補一覧 / トポロジー図 / ΔV 図）と、任意で R/C/L 種別の枝再構築 |
| **Evaluate** | モンテカルロ評価（Top-1/Top-3 精度ほか）。公差・ノイズを掃引した精度曲線プロットにも対応 |

> 故障素子はプルダウンから選ぶため、素子名のタイプミスが起きません。入力エラーは画面上に表示され、アプリは停止しません。

> 📘 **評価の詳しい手順・結果の見方**（CLI / GUI 両対応、図表付き）は **[EVALUATION.md（評価ガイド）](EVALUATION.md)** を参照してください。

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
   * AC（複素 $\Delta V_m$）の場合は、**振幅と位相**の2段グラフを出力します。

---

## ✅ テストの実行

開発やリファクタリング後の完全性を担保するため、`pytest` による回帰テストを完備しています。

```bash
pytest tests/
```
*(回路の回帰テストは厳密なアサーションのため `method='exhaustive'` を、補助モジュールのテストは schema 検証 / YAML ロード / テスタビリティ偽判定 / 評価 / Ridge 再構築をカバーします)*

---

## ⚠️ 制約事項 (Limitations)

* **対応素子は R / C / L**（線形・受動素子）。能動素子やダイオード等の非線形素子、相互インダクタンスには未対応です（PoC のスコープ）。
* **C/L を含む回路は周波数の指定が必須**です（DC では C は開放・L は短絡となり評価できないため、`frequencies` または `--freq` に正の周波数を 1 つ以上指定してください）。
* **OMP は近似解法**です。対称性の高い回路や内部（非アクセス）ノードの故障では取りこぼす場合があるため、小〜中規模では既定の `auto`（自動で全探索）を使用してください。詳細は上記「故障診断アルゴリズムの比較」を参照。
* 診断段の転送インピーダンス行列 `Z_mn` は密行列として保持されるため、超大規模回路ではメモリ・計算量に注意が必要です。

---

## 📚 参考文献 (References)

本ツールが依拠する理論は、以下の論文に基づきます（理論・概念の参照であり、図表・数式・コードの転載は行っていません）。

1. Z. F. Huang, C.-S. Lin, R.-W. Liu, "Node-Fault Diagnosis and a Design of Testability," *IEEE Transactions on Circuits and Systems*, vol. CAS-30, no. 5, pp. 257–265, May 1983.
2. Y. Togawa, T. Matsumoto, "On the Topological Testability Conjecture for Analog Fault Diagnosis Problems," *IEEE Transactions on Circuits and Systems*, vol. CAS-31, pp. 147–158, 1984.

---

## 🧩 サードパーティ・ライブラリ (Third-Party Libraries)

本プロジェクトは、それぞれのライセンス（いずれも BSD / MIT / PSF / Apache-2.0 系の寛容なライセンス）の下で配布される以下のオープンソースライブラリに依存しています。これらのライブラリのコードは本リポジトリに同梱しておらず、`pip` 経由で各自インストールされます。

NumPy, SciPy, scikit-learn, matplotlib, NetworkX, PyYAML（GUI 利用時のみ Streamlit〈Apache-2.0〉、開発用に pytest）

各ライブラリは原著作者・各プロジェクトに著作権が帰属します。詳細は各プロジェクトのライセンスを参照してください。

---

## 📄 ライセンス (License)

本リポジトリは [MIT License](LICENSE) の下で公開されています。

本ソフトウェアは「現状のまま (AS IS)」提供され、明示・黙示を問わずいかなる保証も伴いません。利用に伴う一切のリスクは利用者が負うものとします（詳細は [LICENSE](LICENSE) を参照）。
