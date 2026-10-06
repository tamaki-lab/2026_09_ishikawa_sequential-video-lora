# Sequential Video LoRA

Sequential Loader で動画を時系列順に流し，Q/V LoRA を付けた ViT を Streaming MoCo で
自己教師あり学習して，ActivityNet の Linear Probe で評価する研究コード．
CNN/ViT の教師あり分類テンプレートを土台にしており，`main.py` / `main_pl.py` による
教師あり学習も従来どおり使える．

## ディレクトリ構成

| パス | 役割 |
|---|---|
| `main.py`, `main_pl.py`, `train.py`, `val.py`, `conf/`, `dataset/`, `setup/`, `callback/` | テンプレート由来の教師あり分類（Hydra 設定） |
| `model/` | ネットワーク部品（`backbones/`，`aggregators/`）と分類モデルの factory |
| `self_supervised/moco/` | MoCo の学習則（Query / Key，EMA，Queue，負例選択） |
| `integration/` | `sequential_loader` と利用側の間の処理（時系列 stream，フレーム encode，MoCo の view） |
| `training/` | Streaming MoCo の学習．共通 engine（`streaming_moco.py`），短時間 canary（`moco_canary.py`），本番 full run（`streaming_moco_full.py`），protocol・監査・checkpoint |
| `evaluation/` | ラベルを使う下流評価（manifest，segment feature，Linear Probe） |
| `scripts/` | CLI 入口（`smoke/`，`moco/`，`linear_probe/`）．ルートから `python3 -m scripts....` で実行 |
| `utils/`, `logger/` | 共通 utility，Comet ロガー，成果物の SHA-256・provenance 記録 |
| `test/` | pytest．本体のパッケージ構成に対応（`test/training/`，`test/self_supervised/` など） |

`data/`，`hub/`，`log/`，`comet_logs/` は Git 管理外．

## 準備

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.pytorch.txt
pip install -r requirements.txt

```

## 使い方（教師あり分類）

```bash
python3 main.py dataset=imagefolder dataset.root=/mnt/NAS-TVS872XT/dataset-lab/Tiny-ImageNet/ loader.num_workers=24 loader.batch_size=8 trainer.num_epochs=5 trainer.use_dp=true
python3 main_pl.py dataset=imagefolder dataset.root=/mnt/NAS-TVS872XT/dataset-lab/Tiny-ImageNet/ loader.num_workers=24 loader.batch_size=8 trainer.num_epochs=5 trainer.devices=3
```

`_pl`がついたファイルは Pytorch lightning のコードを使用（ddp のみ対応）．

### multi-GPU 学習

GPU の指定には`CUDA_VISIBLE_DEVICES`を使用すること．

- dp (data parallel) は`main.py`で利用可能
- ddp (distributed data parallel)は lightning の`main_pl.py`で利用可能
  - 注意：複数 GPU を用いる dp や ddp が動作しなくなるため，コード内で GPU 番号を指定するような`torch.device("cuda:0")`は**使わない**．dp や ddp のために，コード内では`torch.device("cuda")`としておく．

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 main.py dataset=imagefolder dataset.root=/mnt/NAS-TVS872XT/dataset-lab/Tiny-ImageNet/ loader.num_workers=24 loader.batch_size=8 trainer.num_epochs=5 trainer.use_dp=true
CUDA_VISIBLE_DEVICES=0,1 python3 main_pl.py dataset=imagefolder dataset.root=/mnt/NAS-TVS872XT/dataset-lab/Tiny-ImageNet/ loader.num_workers=24 loader.batch_size=8 trainer.num_epochs=5 trainer.devices=-1
```

デバッグ用には [launch.json](.vscode/launch.json) を以下のように設定する．

```json
            "env": {
                "CUDA_VISIBLE_DEVICES": "0,1",
            },
```

task 用には[tasks.json](.vscode/tasks.json)に次のように設定する．

```json
            "options": {
                "env": {
                    "CUDA_VISIBLE_DEVICES": "0,1",
                },
            },
```

### 設定と override

学習設定は [conf/config.yaml](conf/config.yaml) を入口に Hydra で合成する．
既定値は `dataset=cifar10`，`model=resnet18`，`optimizer=sgd`．
旧 training CLI の `-d`，`-m`，`-b`，`-lr`，`--devices`，`--scratch` などは廃止した．
単発 utility / smoke script の CLI は従来どおり．Full MoCo と Linear Probe の
実験入口は後述の専用 Hydra config を使う．

| 設定 | 選択肢・例 |
|---|---|
| `dataset` | `cifar10`, `imagefolder`, `videofolder`, `zero_images` |
| `model` | `resnet18`, `resnet50`, `x3d`, `abn_r50`, `vit_b`, `zero_output_dummy` |
| `optimizer` | `sgd`, `adam` |
| `dataset.root` | データセットの root（ZeroImages では不要） |
| `dataset.train_dir`, `dataset.val_dir` | ImageFolder / VideoFolder の split サブディレクトリ |
| `loader.batch_size`, `loader.num_workers` | バッチサイズ，worker 数 |
| `video.frames_per_clip`, `video.clip_duration`, `video.clips_per_video` | clip の frame 数，秒数，検証用 clip 数 |
| `model.use_pretrained=false` | scratch 学習 |
| `model.torch_home` | 事前学習済み weight の保存先 |
| `optimizer.lr`, `optimizer.weight_decay`, `optimizer.momentum` | optimizer 設定（momentum は SGD で使用） |
| `scheduler.enabled=true` | 既存 scheduler を有効化 |
| `trainer.num_epochs`, `trainer.log_interval_steps`, `trainer.grad_accum` | epoch 数，ログ間隔，勾配蓄積 |
| `trainer.val_interval_epochs` | `main.py` の検証間隔（Lightning は従来どおり毎 epoch） |
| `trainer.use_dp=true` | `main.py` の Data Parallel |
| `trainer.devices=0` | Lightning の GPU ID（`-1` は全 GPU） |
| `logging.disable_comet=true` | Comet を無効化 |
| `logging.comet_log_dir` | Lightning の Comet 保存先 |
| `logging.tf_log_dir` | 旧設定の値を保持（既存 training path では未使用） |
| `checkpoint.save_dir`, `checkpoint.resume` | checkpoint 保存先，resume ファイル（未指定は `null`） |

```bash
python3 main_pl.py dataset=imagefolder model=vit_b optimizer=adam loader.batch_size=8 optimizer.lr=1e-4 trainer.devices=0
python3 main_pl.py model.use_pretrained=false scheduler.enabled=true logging.disable_comet=true
# カンマを含む GPU ID は Hydra の文字列として quote する．
CUDA_VISIBLE_DEVICES=0,1 python3 main_pl.py 'trainer.devices="0,1"'
```

`hydra.job.chdir=false` により，データ・weight・checkpoint の相対パスは
起動時の working directory を基準とする．Hydra の実行記録は
`log/hydra/<date>/<time>/` に保存する（Git 管理対象外）．
`.hydra/` に合成設定・override・Hydra 設定を，`main.log` / `main_pl.log` に
補間を解決した実行設定（`Resolved config`）を保存する．独自の YAML copy は行わない．

学習を起動せず，設定と override を確認できる．

```bash
python3 main.py --cfg job --resolve
python3 main_pl.py --cfg job --resolve
python3 main_pl.py dataset=imagefolder model=vit_b optimizer=adam --cfg job --resolve
python3 main.py --help
python3 main_pl.py --help
```

VS Code の [launch.json](.vscode/launch.json) / [tasks.json](.vscode/tasks.json) も
同じ Hydra override 形式を使用する．

## Sequential bridge と smoke チェック

`integration/` に `sequential_loader` と ViT / MoCo をつなぐ共通処理を置く．
手動の動作確認スクリプトは `scripts/smoke/` に置く．リポジトリの
ルートからモジュールとして実行する．

```bash
python3 -m scripts.smoke.smoke_vit_frame_encoder dog.jpg
python3 -m scripts.smoke.smoke_activitynet_vit_clip_feature /path/to/ActivityNet
python3 -m scripts.smoke.smoke_activitynet_vit_lora_moco_multistep /path/to/ActivityNet --max-steps 10
python3 -m scripts.smoke.audit_activitynet_inventory /path/to/ActivityNet
```

従来のルート直下の `smoke_*.py`，`audit_activitynet_inventory.py` は上記のモジュール実行に置き換える．
`import sequential_*_bridge` は互換入口を通して利用できる．
データセット，チェックポイント，Git revision に関する各スクリプトの
検証条件は従来どおり．

## Full-dataset Streaming MoCo と ActivityNet Linear Probe

ActivityNet v1.3 `training` 全 10,024 動画を Adapter 順に 1 回だけ処理する
Stage 6B Streaming MoCo と，annotation segment 単位の Linear Probe の入口．
既存の smoke / canary CLI とは別の明示的な production 入口である．設定の入口は
`conf/moco_full.yaml`，`conf/linear_probe_{manifest,features,run}.yaml`．共通の科学設定は
`activitynet`，`encoder`，`sequential`，`moco`，`linear_probe`，`provenance` の
config group、Comet名などの運用設定は `tracking` group に分離している．

```bash
# 1. MoCo（fresh / resume を明示的に分ける）．出力は log/moco/<run_id>/
python3 -m scripts.moco.train_full_streaming_moco \
    runtime.dataset_root=/path/to/ActivityNet runtime.run_id=<run_id> runtime.seed=<seed> runtime.device=cuda
python3 -m scripts.moco.train_full_streaming_moco \
    runtime.dataset_root=/path/to/ActivityNet runtime.run_id=<run_id> runtime.seed=<same-seed> \
    runtime.device=cuda runtime.resume=true

# 2. 共通 manifest（build の後，再生成 SHA-256 を照合する audit で Gate PASS）
python3 -m scripts.linear_probe.build_manifest \
    runtime.command=build runtime.dataset_root=/path/to/ActivityNet runtime.manifest_id=lp-v1
python3 -m scripts.linear_probe.build_manifest \
    runtime.command=audit runtime.dataset_root=/path/to/ActivityNet runtime.manifest_id=lp-v1

# 3. segment feature（condition ごとに 1 回）
python3 -m scripts.linear_probe.extract_features \
    runtime.dataset_root=/path/to/ActivityNet runtime.manifest_id=lp-v1 runtime.condition=base_vit
python3 -m scripts.linear_probe.extract_features \
    runtime.dataset_root=/path/to/ActivityNet runtime.manifest_id=lp-v1 \
    runtime.condition=moco_query_lora_final \
    runtime.snapshot=log/moco/<run_id>/evaluation_snapshots/<videos-010024_..._final>

# 4. Linear Probe 2 conditions x seeds 0,1,2 と集計
python3 -m scripts.linear_probe.run_probe runtime.manifest_id=lp-v1 \
    runtime.base_features=log/linear_probe/features/base_vit/<feature_id> \
    runtime.lora_features=log/linear_probe/features/moco_query_lora_final/<feature_id>

# 短時間確認（科学設定のoverrideなので自動的にnon-production）
python3 -m scripts.linear_probe.run_probe linear_probe.probe.epochs=1 \
    runtime.manifest_id=lp-v1 runtime.base_features=<base_feature_dir> runtime.lora_features=<lora_feature_dir>

# 実行せずに合成済み設定を確認
python3 -m scripts.moco.train_full_streaming_moco --cfg job --resolve
python3 -m scripts.linear_probe.run_probe --cfg job --resolve

# Comet 登録に失敗・無効だった local artifact の再登録（hash を再検証してから）
python3 -m scripts.retry_comet_artifact log/linear_probe/manifest/lp-v1
python3 -m scripts.retry_comet_artifact log/linear_probe/results/<result_id>/comparison
```

- `log/moco/<run_id>/resume/latest.pt` は 100 動画ごとの video 境界と final で atomic に更新する学習再開用 state．
  `evaluation_snapshots/` は 1,000 動画ごとと final の Query LoRA のみ（PEFT 形式）で，下流評価専用．
- Full MoCo の `runtime.seed` は fresh / resume の両方で明示必須で，同じ値を使う．canonical production は
  untracked file を含む clean checkout だけを受理し，code・seed・device identity の不一致を resume 時に拒否する．
- 既存の run / manifest / feature / result は，内容が一致する場合だけ再利用し，上書きしない．
- `runtime.stop_after_videos`，`runtime.max_videos_per_split`，`linear_probe.probe.epochs` の
  override は短時間 smoke に使える．canonical preset と異なる科学設定の成果物は non-production として記録される．
  実装されていない値（`linear_probe.feature_definition` の変更，scheduler，class weight など）は，
  記録だけが変わるのを防ぐため実行前に拒否する．
- Feature ID と Probe result ID は内容の SHA-256 から自動生成する．任意IDを使う場合だけ
  `runtime.feature_id` / `runtime.result_id` を指定する．データセットやartifactのローカルパスは
  `locations` として記録するが，内容同一性のhashには含めない．
- 全 CLI は resolved config，Git commit / dirty 状態，artifact path，SHA-256，Comet 状態を標準出力または metadata に残す．
  `logging.disable_comet=true` で Comet を使わずに local artifact だけを作る．

## Comet の設定

comet の設定は，

- このディレクトリの`./.comet.config`と，
- ホームの`~/.comet.config`

の 2 つのファイルを利用する．詳しくは[comet のドキュメント](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/advanced/configuration/)を参照．コード中には API キーなどは書かないこと（[logger.py](./logger/logger.py)参照）．

### ホームでの全体設定

- `~/.comet.config`：すべてに共通する設定を書く．
  - comet の API キー，デフォルトの comet workspace を設定．
  - `hide_api_key`は True にすること（しないとログに API キーが残ってしまう）

```ini
[comet]
api_key=XXXXXHereIsYourAPIKeyXXXXXXXX
workspace=tttamaki

[comet_logging]
hide_api_key=True
```

### フォルダごとの設定

- このディレクトリの`./.comet.config`：このディレクトリで使用する設定を書く．
  - comet project name を設定．
  - （ここで設定する内容はホームの`~/.comet.config`よりも優先されて，上書きされる）

```ini
[comet]
project_name=simple_cnn_20230309

[comet_logging]
display_summary_level=0
file=comet_logs/comet_{project}_{datetime}.log

[comet_auto_log]
env_details=True
env_gpu=True
env_host=True
env_cpu=True
cli_arguments=True
```

### コード内での設定

- コード
  - `Experiment`オブジェクトに comet experiment name を設定．必要なら tag を設定する．
  - コード中には **API キーなどは書かない**．
  - （コード中で設定する内容は，ディレクトリごとの`./.comet.config`よりも優先される）

```python
    experiment = Experiment()  # ここでは何も設定しない

    exp_name = datetime.now().strftime('%Y-%m-%d_%H:%M:%S:%f')  # これは日時をexperiment nameに設定する例．
    experiment.set_name(exp_name)
    experiment.add_tag(cfg.model.name)  # これはモデル名をタグに設定する例．
```
