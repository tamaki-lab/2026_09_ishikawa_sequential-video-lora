# A simple CNN/ViT training code

CNN/ViT を使って学習する単純な練習用コードです．

## 準備

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.pytorch.txt
pip install -r requirements.txt

```

## 使い方

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
単発 utility / smoke script の CLI は従来どおり．

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
