# 模型版本与来源

本次初始版本是正式主线 `MarketGate → LLA → PureMLP → scalar readout`，不包含 SGRC、ECA 或 SimAM。模型代码可独立导入，不依赖原服务器的实验目录；训练后的权重和数据不随 Git 上传。

## 结构约定

输入为 `[B,8,221]`，前 158 列是股票特征，后 63 列是市场特征，不含标签。输出为 `[B,1,1]`。

| 部件 | 实现 | 参数量 |
|---|---|---:|
| MarketGate | `2 × sigmoid(M @ W)`，`W[63,158]` 零初始化，无偏置 | 9,954 |
| LLA | 因子级有界连续位移、逐日共享表示、残差融合 | 61,760 |
| PureMLP | 历史维度 `8 → 256`，两层 `256 → 256`，输出 `256 → 1` | 134,145 |
| 读出头 | `Linear(158,1,bias=False)`，初始权重为 `1/158` | 158 |
| 合计 | 所有部件联合训练 | 206,017 |

LLA 保留原正式实现：`tau = 3*tanh(raw_tau)`，每个因子一个全局可学习位移；长度 8 的历史窗口零填充到 16 后进行 FFT；共享表示维度为 32；因子残差强度初始为 0.1。这里并没有额外预测每只股票、每一天的最佳作用时间，LLA 也不只是单独的移位算子，它还包含共享表示与非线性融合。

市场门控初始化为恒等映射，不意味着整个 LLA 也初始化为恒等映射。输入窗口必须只包含预测时点已知的历史信息。

PureMLP 的 `use_tq=False`、`channel_aggre=False`，`temporalQuery` 与 `channelAggregator` 均从模型移除。`cycle=7`（CSI300）与 `cycle=3`（CSI800）作为配置兼容字段保留，`cycle_index` 不参与本版本前向计算。

## 初始化及 checkpoint 兼容性

为了保留原主线的初始化，构造时仍先创建完整上游核心与读出头，再删除禁用部件。直接跳过这些部件的构造会改变随机数消耗顺序。LLA 在 `torch.random.fork_rng(devices=[])` 中初始化，既使用当前种子的随机状态，又不推进外层 CPU RNG。

模型保留原来的 `market_gate_weight`、`core.*`、`readout.weight`、`lla_block.*` checkpoint 键。可使用 `model.load_state_dict(state_dict, strict=True)` 载入原主线的 state dictionary；不需要载入旧服务器上的 Python 模型对象。

## 源码快照

以下均为来源项目内的相对位置，仅用于追溯，不是运行时依赖。

- 上游核心：`TQNet`，提交 `15e19cb23483ed52398566c4baa959168cfffa57`，`models/TQNet.py`。原文件 SHA-256：`c31e4498104d241a69335d22522f01a28cae90d58d73f4ec4c10929fb53b7727`。本仓库 `src/qtmaster/vendor/tqnet.py` 的可执行代码不变，仅规范化末尾空行，SHA-256：`3d19da0236ee289563b12a8e23f2ddd8f2aff6aea651c58284a391762ab268f5`。
- CSI300 正式主线：`MASTER-master/runs/tqnet_market63_gate_lla_pure_mlp_csi300_gpu45_pkg_20260830T122331Z/tqnet_stock_runner.py`，SHA-256：`88529f4c676165062793a728588866af33787cb47d476edaee3198a1334668ff`。
- CSI800 正式主线：`MASTER-master/runs/tqnet_market63_gate_lla_pure_mlp_csi800_direct_gpu45_pkg_20260830T124943Z/tqnet_stock_runner.py`，SHA-256 同上。
- LLA：上述 CSI300 正式主线目录的 `lead_lag.py`，原样导出到本仓库 `src/qtmaster/lead_lag.py`，SHA-256：`d6fea3a39d12f79b0fde448ca189221fb3fa82d6bb10d2a39fc75c903087f558`。
- 冻结 CSI300 构造器：`MASTER-master/runs/tqnet_dmodel256_p5e30_gpu1_20260828T142052Z/csi300/code/tqnet_stock_runner.py`，SHA-256：`7be6ab37f55f42995b38b375b10c38c4d532aa0d76850c2dfa290f9b4b71c7e4`。
- 冻结 CSI800 构造器：同一来源快照的 `csi800_direct/code/tqnet_stock_runner.py`，SHA-256：`a120659f74cc3640ac3a1b570c63e40b135de301fbcffbaf3d94b7bb13785b44`。

上游 TQNet 的 Apache-2.0 许可全文保留在 `licenses/TQNet-Apache-2.0.txt`，归属见 `NOTICE`。该许可文件只声明相应第三方部件的来源与许可，不代表为整个 QtMaster 项目新增统一许可。

## 导出检查

2026-09-06，在原 PyTorch 2.1.2 CPU 环境中实际加载上述两个正式主线构造器，并分别用种子 0–4 构造原模型和本仓库模型，共完成 10 组对照：

- 21 个 `state_dict` tensor 的键、顺序与值完全一致；
- 同一输入的前向输出与全部参数梯度逐元素完全一致；
- 构造完成后的 CPU RNG 状态完全一致；
- 参数量均为 206,017。

独立测试 `tests/test_model.py` 另外覆盖输出形状、门控初始恒等、有限梯度、checkpoint 回载、初始化 RNG 和拒绝标签列。测试不依赖服务器、数据文件或旧模型目录：

```bash
python -m unittest discover -s tests -p test_model.py -v
```

上述检查确认本次代码导出没有改变模型，不承诺不同设备或软件版本上的完整训练结果逐位一致。
