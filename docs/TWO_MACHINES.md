# 两台机器各自实验

`main` 保存首次可运行的共同起点。两台机器不需要同步各自的后续改进，也不要求结果逐位相同。

| 分支 | 用途 |
| --- | --- |
| `main` | 共同起始代码；日常实验不向这里推送 |
| `exp/server` | 当前服务器的独立改进 |
| `exp/local` | 另一台电脑的独立改进 |

## 当前服务器

首次上传后，服务器的独立代码仓库位于 `QtMaster` 目录，工作分支为 `exp/server`。
原有实验目录仍保留，不会自动变成这个仓库的一部分。
之后在这份 Git 仓库中修改代码、保存配置，再推送到对应分支：

```bash
git branch --show-current
git status --short
git add src/ configs/
git commit -m "Describe the server experiment"
git push origin exp/server
```

只暂存本次需要的文件；不要把密钥、数据或完整运行目录加入提交。

## 另一台电脑

下载公开仓库可以用 HTTPS。若仓库为私有，先配置自己的 GitHub 访问权限。
需要上传时，在该电脑生成自己的 SSH key，将公钥添加为该仓库的可写 Deploy Key；
不要复制服务器私钥。配置见 [GitHub Deploy Keys 文档](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/managing-deploy-keys)。

```bash
git clone https://github.com/dorightthings/QtMaster.git
cd QtMaster
git switch -c exp/local origin/main
# 配好该电脑自己的 SSH key 后启用 SSH 推送：
git remote set-url origin git@github.com:dorightthings/QtMaster.git
git push -u origin exp/local
```

如果远程已经存在 `exp/local`，改用 `git switch --track origin/exp/local`，不要重复创建。
环境与数据配置按根目录 README 进行。

## 各自保存进展

```bash
git branch --show-current
git status --short
git add src/ configs/ docs/
git commit -m "Describe the local experiment"
git push origin exp/local
```

两端默认不合并。某个改进有效时，只推到它所在的分支。
确实需要借用另一个分支的改进时再手动 cherry-pick 或合并，不设置自动同步或自动合并。

数据使用同一个 `data/manifest.json` 校验版本。checkpoint、预测和完整日志留在各自机器；
需要分享结果时，可以单独提交不含本机路径和敏感信息的小型 CSV/Markdown 摘要。
