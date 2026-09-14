# Git 工作流

> 仓库范围：**只跟踪 `when/`**，其余目录（EyeWO/、MD/、视频、PDF）全部不进版本库。
> 提交说明和文件名一律用**英文**，避免终端或其他工具渲染不出中文。

---

## 一、日常四步

```bash
# 1. 改代码之前，先确认当前是干净的
git status --short          # 没输出 = 干净，可以放心改

# 2. 改完、跑过、满意了 —— 存档（说明用英文）
git add when/
git commit -m "Tune cooldown to 5s for rapid repeated actions"

# 3. 如果这一版值得以后回来 —— 打版本号
git tag -a v1.1 -m "Cooldown 20s -> 5s; repeated pickups now trigger"

# 4. 随时查历史
git hist
```

**唯一的铁律：改之前先 commit。** 只要满意的版本都提交过，就永远回得去。

---

## 二、提交说明写法

全英文，首字母大写，用动词开头的祈使句，不加句号。

```bash
# 好
git commit -m "Add per-query threshold calibration"
git commit -m "Fix buffer aliasing in audio callback"
git commit -m "Lower cooldown to 5s for rapid repeated actions"

# 不好
git commit -m "更新"           # 中文，可能渲染不出
git commit -m "fixed some bugs"  # 说了等于没说
```

常用前缀（可选，但推荐）：

| 前缀       | 用途           |
| ---------- | -------------- |
| `Add`      | 新功能         |
| `Fix`      | 修 bug         |
| `Tune`     | 调参数         |
| `Refactor` | 重构，行为不变 |
| `Remove`   | 删功能         |
| `Docs`     | 只改文档       |

---

## 三、版本号约定

用 `v主版本.次版本`：

| 什么时候升               | 例子                     |
| ------------------------ | ------------------------ |
| **次版本** `v1.0 → v1.1` | 调参数、修 bug、小改动   |
| **主版本** `v1.x → v2.0` | 换算法、改架构、接口变了 |

tag 说明要写**这一版的特征和实测结果**，以后你是靠这句话找回它的：

```bash
git tag -a v1.1 -m "Cooldown 20s -> 5s; 4 sq registered live, repeated pickups all fire"
```

---

## 四、回滚

```bash
# 最常用：把代码恢复到某一版，不动历史
git checkout v1.0 -- when/

# 恢复错了想撤销
git checkout HEAD -- when/

# 只恢复某一个文件
git checkout v1.0 -- when/gate.py

# 丢弃所有未提交的改动
git checkout -- when/
```

---

## 五、查历史

```bash
git hist            # 哈希 + 时间 + 版本号 + 说明（一行一版）
git versions        # 只看打过版本号的
```

`git hist` / `git versions` 是本仓库配好的快捷命令，存在 `.git/config`。

### 看差异

```bash
git diff                        # 当前改了什么还没提交
git diff v1.0                   # 某版 vs 现在
git diff v1.0 v1.1              # 两版之间
git diff v1.0 -- when/gate.py   # 只看某个文件
```

---

## 六、问题速查

| 情况                   | 命令                                  |
| ---------------------- | ------------------------------------- |
| 改乱了，想回到上次提交 | `git checkout -- when/`               |
| 回到某个满意的版本     | `git checkout v1.0 -- when/`          |
| 看某版本当时的某个文件 | `git show v1.0:when/gate.py`          |
| 提交说明写错了         | `git commit --amend -m "New message"` |
| 看某个文件的修改历史   | `git log --oneline -- when/gate.py`   |
| 版本号打错了           | `git tag -d v1.1` 后重新打            |
| 确认跟踪了哪些文件     | `git ls-files`                        |

---

## 七、仓库范围

分两层，各管各的：

**第一层 —— 仓库范围**（`.git/info/exclude`，仓库本地配置，不进版本库）

```
/*
!/when/
```

先排除根目录全部内容，再放行 `when/`。所以 `EyeWO/`、`MD/`、PDF、`diagram.png`
既不会被跟踪，也不会在 `git status` 里刷屏。

> 为什么不放在 `when/.gitignore` 里：子目录的 `.gitignore` 只对该目录内部生效，
> 管不到根目录。范围规则必须放在仓库级别。

**第二层 —— when 内部**（`when/.gitignore`，**进版本库**）

```
__pycache__/   *.py[cod]     编译缓存
*.jsonl  outputs/            运行产物
clips/  *.mov  *.mp4         视频（各 77MB）
.DS_Store
```

`when/clips/` 里的两段视频仍在硬盘上，只是不被跟踪。

**文件名也要用英文**。git 对非 ASCII 文件名会显示成 `"GIT\345\267\245..."` 这种转义，
跟中文提交说明是同一个问题。本仓库已设 `core.quotepath false` 作为兜底。

---

## 八、版本记录

| 版本     | 哈希      | 说明                                                                                                                                          |
| -------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **v1.0** | `f4d5e6c` | WHEN 基线。live 语音注册 4 条 standing query 实测通过（11 次 STANDING 触发、10 次正确）；逐条阈值标定（anchored to chance level）；20s 冷却期 |

回到某一版：

```bash
git checkout v1.0 -- when/
```
