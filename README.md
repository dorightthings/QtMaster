# QtMaster

当前可独立训练的市场条件化时间校准模型，用作新机器的开发起点。
本分支只提供这一种模型，不包含历史实验版本、对比方法、实验成绩或旧权重。

```text
历史股票因子（158维）＋历史市场特征（63维）
                    ↓
             市场条件因子重标定
                    ↓
      当天市场状态驱动的因子级时间校准
                    ↓
             PureMLP＋线性预测头
                    ↓
                 收益预测
```

输入为过去8个交易日的221维特征，不含标签；输出每只股票一个分数。
模型共有 **213,279** 个可训练参数。所有部分联合训练，使用同一个Adam参数组。
结构和来源见 [模型说明](docs/MODEL_PROVENANCE.md)。

## 下载当前版本

```bash
git clone --branch exp/local --single-branch https://github.com/dorightthings/QtMaster.git
cd QtMaster
```

请显式选择 `exp/local`；`main` 不作为此次交付入口。

## 安装项目与检查

目标电脑的Python、PyTorch及GPU环境另行配置。本次没有安装或验证5070 Ti环境。
`requirements.txt` 保留项目参考依赖；不要将它当作适用于所有显卡的完整环境锁文件。
在已配好的Python/PyTorch环境中安装本项目：

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
python -m qtmaster.train --config configs/csi300.yaml --smoke-test --seeds 0
```

`--smoke-test` 只在CPU上用合成数据跑两轮，检查训练、验证、checkpoint和预测保存。
它不使用真实数据、不执行金融回测，也不产生可用于模型评价的金融指标。

## 放置数据

参见 [数据说明](data/README.md)。两个股票池使用同一个 `data/manifest.json` 校验文件身份。
下载链接目前尚未填写；取得可信数据包前只能运行合成检查。
完整的六项指标还需要匹配的Qlib行情provider。

```text
data/
├── manifest.json
├── csi300/
│   ├── train.pkl
│   ├── valid.pkl
│   └── test.pkl
├── csi800_direct/
│   ├── train.pkl
│   ├── valid.pkl
│   └── test.pkl
└── qlib_provider/
    ├── calendars/day.txt
    ├── instruments/
    └── features/
```

运行入口只依赖当前代码、指定数据和可选provider，不要求存在其他项目、服务器路径、
历史结果目录或任何旧checkpoint。

## 从头训练和回测

以下命令每次都创建新的运行目录，种子顺序执行，不覆盖已有运行：

```bash
python -m qtmaster.train --config configs/csi300.yaml --data-root data --output-root outputs --device cuda:0 --seeds 0 1 2 3 4 --trust-pickle --qlib-provider data/qlib_provider
python -m qtmaster.train --config configs/csi800_direct.yaml --data-root data --output-root outputs --device cuda:0 --seeds 0 1 2 3 4 --trust-pickle --qlib-provider data/qlib_provider
```

只跑一个种子时使用 `--seeds 0`。不指定 `--seeds` 时默认0–4。
如果暂不回测，去掉 `--qlib-provider`；此时计算IC、ICIR、RankIC和RankICIR，
AR、IR标记为未计算，不会填写为0。
`--trust-pickle` 仅用于自己确认来源可信的数据。

## 当前配置

| 项目 | CSI300 | CSI800-Direct |
| --- | --- | --- |
| 初始学习率 | 0.001 | 0.0005 |
| batch / 评估batch | 64 / 64 | 64 / 64 |
| 最大epoch / patience | 30 / 5 | 30 / 5 |
| 优化器 | Adam，单参数组 | Adam，单参数组 |
| 学习率调度 | type3，epoch结束后更新 | type3，epoch结束后更新 |
| 数据加载worker | 2 | 2 |

训练丢弃不足batch的最后一批；验证和测试遍历全部样本。
checkpoint按完整验证集有限标签的SSE/count最小值选择，相等时采用后一个。
测试仅评估选中的checkpoint，不用于选择训练轮次。

数据保留当前 `no_purge` 划分和标签处理，不在迁移过程中重新切分或重建因子。
回测采用Top30/Drop30；AR、IR为无成本超额收益口径，模拟器成本和换手记录另存。
每日IC相关比率使用总体标准差；当前单次运行的种子汇总也明确记录 `std_ddof=0`。

## 目录与后续开发

- `src/qtmaster/`：当前模型、数据读取、训练、指标和回测。
- `configs/`：两个股票池的起始设置。
- `scripts/export_data_bundle.py`：可信源数据的校验与打包工具。
- `tests/`：无需旧模型、旧实验目录的独立检查。
- `outputs/`：新机器生成的训练日志、预测、checkpoint和结果，不进入Git。

日常提交只包含代码、配置和说明。分支使用见 [开发方式](docs/TWO_MACHINES.md)。
第三方来源与许可保留在 `NOTICE`、`licenses/`；数据不随Git分发。
