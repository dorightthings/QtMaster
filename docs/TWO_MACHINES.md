# 新电脑开发方式

仓库默认的 `main` 是当前模型的独立开发起点，
工作目录不包含服务器历史实验版本、历史结果或旧权重。

## 下载

```bash
git clone --depth 1 https://github.com/dorightthings/QtMaster.git
cd QtMaster
git branch --show-current
```

预期分支为 `main`，不需要再创建或切换实验分支。
`--depth 1` 只取得当前提交，后续仍可正常开发和推送。
配置环境和取得数据后，按README从头训练。

## 保存改进

```bash
git status --short
git diff
git add src/ configs/ tests/ README.md docs/
git diff --cached
git commit -m "Describe the model change"
git push origin main
```

只暂存本次需要的文件；新增文件可以单独指定。
本机生成的 `outputs/`、数据、checkpoint、私有路径配置和凭据不提交。
使用新电脑自己的GitHub认证，不复制服务器私钥。

日常改进直接推送到 `main`；若远程有新提交，先检查并整合，不使用强推覆盖。
服务器原工作目录和实验记录独立保留，不要求跨机器结果一致。
