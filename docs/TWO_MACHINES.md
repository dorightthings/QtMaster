# 新电脑开发方式

本次交付分支为 `exp/local`。它是当前模型的独立开发起点，
工作目录不包含服务器历史实验版本、历史结果或旧权重。

## 下载

```bash
git clone --branch exp/local --single-branch https://github.com/dorightthings/QtMaster.git
cd QtMaster
git branch --show-current
```

预期分支为 `exp/local`。配置环境和取得数据后，按README从头训练。

## 保存改进

```bash
git status --short
git diff
git add src/ configs/ tests/ README.md docs/
git diff --cached
git commit -m "Describe the model change"
git push origin exp/local
```

只暂存本次需要的文件；新增文件可以单独指定。
本机生成的 `outputs/`、数据、checkpoint、私有路径配置和凭据不提交。
使用新电脑自己的GitHub认证，不复制服务器私钥。

`main` 与服务器的 `exp/server` 本次不变；日常改进推送到 `exp/local`。
不存在自动同步、自动合并或跨机器结果一致性要求。
