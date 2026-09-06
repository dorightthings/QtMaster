# 数据下载与目录

下载链接：待补充。`manifest.json` 中的 `download_url` 暂为 `null`。
当前 Git 仓库不包含数据；只下载代码可以运行合成数据自检，真实训练需要下列数据包。

解压到仓库根目录后应得到：

```text
data/
  manifest.json
  csi300/{train,valid,test}.pkl
  csi800_direct/{train,valid,test}.pkl
  qlib_provider/                 # 可选，只有收益回测需要
    calendars/day.txt
    instruments/...
    features/...
```

原始窗口为 float32 `[8,222]`：前 158 列为个股因子，后 63 列为市场信息，
最后一列为标签。模型只接收前 221 列；整个历史窗口内的标签列都会排除。
训练集按日排除非有限标签，各去掉 `floor(有限标签数 × 2.5%)` 个最高、最低标签，
对保留样本按 `ddof=1` 标准化。验证和测试不去尾，按日对有限标签标准化，缺失标签保留。
这延续当前开发实验的 `no_purge` 划分，不额外修改时间边界。

| 数据集 | 训练原始样本 | 验证样本 | 测试样本 |
|---|---:|---:|---:|
| CSI300 | 856,246 | 17,700 | 183,300 |
| CSI800-Direct | 1,757,732 | 47,200 | 488,721 |

训练区间为 2008-01 至 2020-03，验证为 2020-04 至 2020-06，测试为 2020-07 至 2022-12。
CSI800-Direct 保留历史上游 provider 的成分股截断口径，不等同于修正重建后的 CSI800。
准确交易日边界、每份文件大小和 SHA256 见 `manifest.json`。数据来自冻结的
`chenditc/investment_data@2024-12-07` provider；使用及分发前须确认原数据许可。

## 从原机器打包

在有六个原始 pickle 的机器上，运行：

```bash
python scripts/export_data_bundle.py \
  --source-root /path/to/existing-data \
  --output /path/to/new-qtmaster-data.tar
```

`--source-root` 接受已整理好的 `csi300/train.pkl` 等目录，也接受原始
`csi300_author_fix_full_v2/csi300_dl_train.pkl`、
`csi800_author_fix_full_v2/csi800_author_dl_train.pkl` 目录布局。
如位置不统一，使用 `--source-map /path/to/local-source-map.json` 替代 `--source-root`：

```json
{
  "csi300": {"train": "path/train300.pkl", "valid": "path/valid300.pkl", "test": "path/test300.pkl"},
  "csi800_direct": {"train": "path/train800.pkl", "valid": "path/valid800.pkl", "test": "path/test800.pkl"}
}
```

映射中的相对路径以映射文件所在目录为准。不要把含机器路径的本地映射提交到 Git。
需要收益回测时再加 `--provider-root /path/to/complete-qlib-provider`。
导出过程不反序列化数据，只逐文件核对大小和 SHA256，并保留原始字节。
输出必须是一个不存在的新文件，父目录必须已存在；失败的 `.partial` 保留供检查，不覆盖旧文件。
六个 pickle 合计约 3.15 GB（不含可选 provider），归档不进入 Git。

确认包来自自己的机器或其他可信来源后，在另一台机器的仓库根目录解压：

```bash
tar -tf /path/to/qtmaster-data.tar
tar --keep-old-files -xf /path/to/qtmaster-data.tar \
  data/csi300 data/csi800_direct
```

若包同时含 provider，再单独解压 `data/qlib_provider`。`--keep-old-files` 禁止覆盖已有数据；
包中的 `data/manifest.json` 与仓库已跟踪的清单对应，无需重复覆盖。

## Pickle 安全

这些文件是 Qlib `TSDataSampler` pickle，加载依赖 `pyqlib`。**Pickle 可以执行代码**；
只使用自己导出或可信来源提供的数据，训练时显式传入 `--trust-pickle`。
程序会先按仓库清单校验三个 split 的大小和 SHA256，再反序列化。
哈希只能确认文件身份，不能让不可信的 pickle 变安全；不要随下载包一起盲目信任被替换的清单。
