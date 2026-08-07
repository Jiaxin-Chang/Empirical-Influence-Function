# 从两段代码生成 visualizer 可视化

给两个已经算好 token embedding 的样本，直接在 TTAV 里看它们的 token 分布。
**不经过归因报告、不加载模型、不需要 GPU。**

对应脚本：[`export_pair_bundle.py`](export_pair_bundle.py)

---

## 一、拉代码

```bash
git clone git@github.com:Nishinan/Empirical-Influence-Function.git
cd Empirical-Influence-Function
git checkout feat/inline-plot-and-pair-bundle
```

已经 clone 过的话：

```bash
git fetch origin
git checkout feat/inline-plot-and-pair-bundle
git pull origin feat/inline-plot-and-pair-bundle
```

> `origin` 是我们的 fork（`Nishinan/*`），`upstream` 才是上游（`code-philia/*`）。

TTAV 仓库有同名分支，但**只有部署服务器的人需要**，跑脚本用不到。

## 二、环境

只要三个包，**不需要 GPU、不需要下载模型**——`torch` 只用来读 `.pt`，不做推理，装 CPU 版就够：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy scikit-learn
```

## 三、输入数据

每个样本一个 `.pt`，需要这些字段：

脚本**必需**的字段（缺任何一个会报 `missing required field(s)`）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `hidden` | `[n_tokens, dim]` | 最后一层 token embedding |
| `input_ids` | `[n_tokens]` | |
| `labels` | `[n_tokens]` | `-100` 是 prompt，其余是答案区 |
| `token_surfaces` | `list[str]` | 已解码的 token，如 `' are'`（不是 `Ġare`） |
| `layer` | `int` | 取的第几层 |
| `model_name_or_path` | `str` | 模型标识 |

可选：`adapter_path`（会记进 bundle 的 `checkpoint_path`）、`text`（脚本不读，
但留着便于人工核对）。

`hidden`、`input_ids`、`labels`、`token_surfaces` 四者长度必须相同，否则报错。

**两个文件必须同模型、同层**，否则脚本硬报错——跨嵌入空间求余弦会得到看着正常
但毫无意义的数字，所以这里不给警告、直接失败。

配套的 `.txt` 不用传。

## 四、跑

```bash
python -m src.export_pair_bundle \
    --a /你的路径/000000.pt \
    --b /你的路径/000001.pt \
    --ttav-url http://1.94.115.154/ \
    --open
```

路径随便放哪都行，脚本对目录零假设。

跑完打印一个链接，`--open` 会直接用默认浏览器打开；没打开就复制那行（已做
URL 编码，可以双击整段选中）：

```
Open in browser:
http://1.94.115.154/?eif_jump=%7B%22source%22%3A%22eif%22...
```

### 常用参数

```bash
--projection umap          # 换降维方法，默认 pca
--sample-id my_exp_01      # 自定义名字，默认由文件名+路径哈希生成
--no-upload --out-dir /tmp # 只生成不上传，先看结构
```

### 耗时

数组自动压成 float16 传输，**无损**（原始 embedding 本来就是 bfloat16 推理产物，
实测往返误差为 0）。3839 token × 4096 维的样本对约 **31 MB**，服务器本地实测
2.8 秒；从自己电脑传取决于上传带宽。

---

## 常见问题

**`413 Request Entity Too Large`**
走了内联 JSON 的老路径。确认没手动加 `--array-transport inline`，默认的
`binary` 不会有这个问题。

**`model mismatch` / `layer mismatch`**
两个 `.pt` 不是同模型同层生成的，需要重新导出。

**`missing required field(s)`**
`.pt` 缺字段，对照第三节的表检查。

**点标签前面多一个数字（如 `3588.BP1059`）**
浏览器缓存了旧版前端，`Ctrl+Shift+R` 强制刷新。

**页面打不开或图是空的**
先确认 TTAV 服务活着：浏览器打开 `http://1.94.115.154/` 应该能看到界面。

---

## 图怎么读

会看到两团点，分别是两个样本的 token，按 prompt / 答案区着色（4 类）。

**如果两段代码不相关，它们在图上明显分开是正常的。** 实测：归因算法配对出来的
样本，kNN 同侧近邻占比 0.53（≈完全混合，随机基线 0.53）；两个任意样本是 0.85
（明显分离）。原因是配对样本共享 30-47% 的 prompt 模板，任意两个样本只共享
2-6%。

**目前不画连线**，因为两个任意样本之间没有归因关系，不存在既定的 source→target
边。要基于相似度自动连线得先定义连线规则，那是下一步。

---

## 附：部署 TTAV 服务才需要看

**共用服务器（`1.94.115.154`）已经配好，跑脚本的人不用管这节。**

二进制上传接口 `/registerEIFBundleArray` 要传 30MB 量级的 `.npy`，nginx 默认配置
有两个障碍：`client_max_body_size` 太小会 413；默认会把整个上传缓冲到临时文件才
转发，白白多一次磁盘往返，也失去流式写盘的意义。

在 nginx 的 server 块里加：

```nginx
# location = 是精确匹配，优先级高于下面的正则 API 块。
# 放在通用 API 块之前，是为了让人读配置时一眼看到这个特例。
location = /registerEIFBundleArray {
    client_max_body_size 1g;      # 留余量，当前实际约 31MB
    proxy_request_buffering off;  # 关键：让后端边收边写盘，常数内存
    proxy_read_timeout 900s;
    proxy_send_timeout 900s;
    proxy_pass http://127.0.0.1:5050;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
```

**没有动全局的 `client_max_body_size`**，其他接口不受影响。

应用：

```bash
sudo cp /etc/nginx/conf.d/ttav.conf /etc/nginx/conf.d/ttav.conf.bak.$(date +%Y%m%d%H%M%S)
sudo nginx -t          # 语法报错就别 reload
sudo systemctl reload nginx
```

验证：

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST \
  "http://你的服务器/registerEIFBundleArray?sample_id=probe_test&kind=embeddings" \
  --data-binary "not-npy"
```

期望 **400**（内容不是合法 `.npy`，被后端正确拒绝，说明请求穿过了 nginx）。
返回 404 是没转发给后端；返回 413 是体积限制没生效。

> ⚠️ 这段 nginx 配置**不在任何 git 仓库里**。新部署一台机器必须手动加，否则二进制
> 上传报 413，而报错信息不会有任何线索指向 nginx。值得把它存进 TTAV 仓库的
> `deploy/` 下，目前还没做。

前端 `web/dist` 也是 gitignore 的，拉了代码要自己 `npm run build`，否则看到的是
旧行为。
