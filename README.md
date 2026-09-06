# QtMaster

用于在不同机器上开展独立选股实验的主线代码。首次版本为：

```text
股票历史因子（158维） + 市场特征（63维）
                    ↓
          市场条件因子重标定（MarketGate）
                    ↓
            相对时间校准模块（LLA）
                    ↓
             PureMLP + 线性预测头
                    ↓
                 收益预测
```

输入是过去 8 个交易日的特征，输出每只股票一个分数。总可训练参数为 **206,017**。
这个初始版本不含 SGRC、ECA、SimAM，也不把已有实验目录或 checkpoint 打包进代码仓库。

LLA 使用训练得到的因子级有界相对偏移，并结合逐日共享表示和残差融合；
这些偏移不是对每只股票、每天单独预测的“最佳生效时间”，也不是因果时滞结论。
模型来源和数学实现见 [模型说明](docs/MODEL_PROVENANCE.md)，交付前实际检查见
[验证记录](docs/VALIDATION.md)。

## 1. 下载和配置环境

```bash
git clone https://github.com/dorightthings/QtMaster.git
cd QtMaster
conda env create -f environment.yml
conda activate qtmaster
```

然后安装适合本机的 PyTorch。原实验使用 PyTorch 2.1.2；支持 CUDA 12.1 的 NVIDIA
环境可使用下面的官方历史版本命令：

```bash
python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install -e .
```

仅检查 CPU 运行时，可以把上面的 PyTorch 索引换成 `https://download.pytorch.org/whl/cpu`。
其他显卡或较新的硬件请按 [PyTorch 安装说明](https://pytorch.org/get-started/previous-versions/)
选择兼容驱动的版本，不必强求两台机器的 CUDA 环境相同。
`environment.yml` 只创建 Python 环境；不要跳过后面的依赖安装步骤。

## 2. 先做不需要数据的检查

```bash
python -m unittest discover -s tests -v
python -m qtmaster.train --config configs/csi300.yaml --smoke-test --device cpu --seeds 0
```

合成数据 smoke test 仅检查训练、验证、checkpoint 与输出流程，不能作为股票实验结果。

## 3. 下载数据

数据下载链接由仓库所有者稍后填写，见 [data/README.md](data/README.md)。
**当前 Git 仓库不包含真实数据；在链接补齐或手动转移数据包之前，只能运行合成检查。**

两个股票池必须使用 `data/manifest.json` 中对应的数据文件，不能把不同版本的数据混用。
只加载自己确认可信的数据包；pickle 即使通过哈希校验也不适合接收不可信来源。

回测 AR/IR 还需要与这批数据相匹配的 Qlib 日频行情 provider。
未提供 provider 时仍可训练并计算 IC、ICIR、RankIC、RankICIR，但 AR/IR 会标为未计算。

## 4. 正式训练

```bash
python -m qtmaster.train --config configs/csi300.yaml \
  --data-root data --output-root outputs --device cuda:0 \
  --seeds 0 1 2 3 4 --trust-pickle

python -m qtmaster.train --config configs/csi800_direct.yaml \
  --data-root data --output-root outputs --device cuda:0 \
  --seeds 0 1 2 3 4 --trust-pickle
```

要同时完成回测，在命令后添加 `--qlib-provider data/qlib_provider`。
这里的 `cuda:0` 是当前进程可见的第 0 张卡。例如仅开放物理 GPU 2：

```bash
CUDA_VISIBLE_DEVICES=2 python -m qtmaster.train --config configs/csi300.yaml \
  --data-root data --output-root outputs --device cuda:0 --seeds 0 --trust-pickle
```

同一个命令中的多个 seed 顺序执行；并行实验可以使用不同 GPU 启动独立命令。
所有运行写入新的输出目录，不覆盖已完成的结果。代码会保存配置、环境信息、seed、
训练记录、验证选中的 checkpoint、预测和结果汇总；这些产物默认不进入 Git。

## 固定起始训练协议

| 项目 | 设置 |
| --- | --- |
| 历史窗口 / 模型输入 | 8 日 / 158 股票因子 + 63 市场特征 |
| 优化器 / 初始学习率 | Adam 单参数组 / 0.001 |
| batch / train drop_last | 64 / true |
| 最大 epoch / patience | 30 / 5 |
| 学习率调度 | 原 type3；epoch 结束后更新 |
| checkpoint | 完整验证集有限标签的 MSE 最小值；相等时采用后一个 |
| 测试集 | 仅评估最终选中的单一 checkpoint，不用于选 epoch |
| 训练标签 | 每日去缺失与两端各 floor(2.5%)，剩余样本 ddof=1 标准化 |
| 验证、测试标签 | 每日 ddof=1 标准化，不截尾，保留缺失标签的预测行 |
| 回测 | Top30 / Drop30；头部 AR/IR 是不含交易成本的超额收益指标 |

初始配置与原主线一致；可移植入口简化了服务器专用调度与缓存。
不同机器、依赖或数据加载执行方式可能产生数值差异，不承诺逐位复现历史结果。
跨机器实验的共同起点由代码 commit、数据 manifest 和配置共同确定。

## 两台机器各自开发

`main` 作为共同起点；当前服务器使用 `exp/server`，另一台机器使用 `exp/local`。
后续各自改进只推送到自己的分支，不自动合并到 `main`。
详细步骤见 [两台机器工作流](docs/TWO_MACHINES.md)。

## 来源与权利

第三方 PureMLP 核心的来源、修改说明与 Apache-2.0 文本保留在 `NOTICE`、`licenses/`
及模型说明中。第三方许可证不自动代表仓库所有者对全部新增代码授予同一许可证。
数据不随代码发布，使用与分享数据前应确认自己的数据访问和转授权权限。
