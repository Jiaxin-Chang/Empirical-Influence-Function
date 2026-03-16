# Deprecated / Archived Files

本目录存放所有不在当前实验主流程中的历史代码。  
这些文件均**不应被直接运行或引用**，仅作研究演进记录保留。

---

## 当前主流程（`src/` 根目录）

| 文件 | 作用 |
|---|---|
| `NIF.py` | 核心库：模型加载、数据集构建、infer、saliency 计算 |
| `loss.py` | 所有 loss / gradient / saliency 计算函数 |
| `process_data.py` | ChatML 格式转换、CustomCollator |
| `auto_annotate.py` | GPT API 自动标注训练样本中的关键 token |
| `intervention_experiment.py` | Stage 1 实验：产出 `correlation_matching_results.json` |
| `saliency_intervention_experiment.py` | Stage 2 实验：因果干预验证，产出 `causal_intervention_results.json` |

---

## 本目录文件说明

### 🔴 完全废弃（概念验证阶段遗留）

#### `IF.py`
**最初版本的 Influence Function 实现。**  
- 面向图像分类任务（CIFAR-10 / ResNet18），与当前 LLM 场景无关
- 实现了三种 IF 变体：
  - `BaseInfluenceFunction`：标准 IF，含逆 Hessian 估计（LiSSA 算法）
  - `EmpiricalIF`：经验 IF，用梯度下降/上升模拟扰动，计算 test-train loss 联动
  - `TracIn`：基于梯度内积的 TracIn（Pruthi et al.）
- 当时的 `__main__` 部分跑的是 CIFAR-10 测试
- **依赖** `utils.py`

#### `utils.py`
**配套 `IF.py` 的底层数学工具。**  
- `calc_loss()`：批量计算 loss
- `grad_loss()`：批量计算梯度
- `inverse_hessian_product()`：LiSSA 算法估计 H⁻¹v
- `hessian_vector_product()`：双重反向传播计算 Hv

---

#### `IF_HF.py`
**第二版 EIF 实现，面向 LLM (HuggingFace)。**  
- 包含针对 Qwen/Llama 类模型的 `EmpiricalIF` 类，已适配 Accelerate 多卡
- 实现了两种 influence 计算策略：
  - `query_influence()`：在 test query 上梯度下降，观察训练集 loss 变化
  - `query_resonance_influence()`：双向条件共振探测（Stage A: Q 驱动，Stage B: 条件探测）
- 内嵌了大量 HTML 报告生成逻辑（`save_query_report_html`、`save_query_report_html_attn`），可生成交互式 attention heatmap 报告
- **已被 `NIF.py` 完全替代**
- 注意：import 路径为旧式相对路径（`from process_data import *`），无法在当前包结构下直接运行
- **依赖** `attribution.py`、`vis.py`、`utils.py`（均在本目录）

#### `NIF_old.py`
**`NIF.py` 的旧版存档（New Inference Function）。**  
- 当前 `src/NIF.py` 的前一个主要版本
- 保留以便追溯功能演进 diff

#### `attribution.py`
**早期 saliency / attention attribution 实现。**  
- 实现了两类归因方法：
  - **Attention Attribution**：提取最后一层 attention 权重
    - `attention_attribution_static()`：对 ground-truth token 的静态 attention
    - `attention_attribution_on_generation()`：逐步生成时的动态 attention
  - **Gradient Saliency**：Input × Gradient（∂logit/∂embedding）
    - `gradient_saliency_static()`：单 token 静态 saliency
    - `gradient_saliency_on_generation()`：生成序列的逐步 saliency
- 使用了 `model.set_attn_implementation('eager'/'sdpa')` API，已被 `loss.py` 中更精确的二阶 saliency 实现取代

#### `vis.py`
**配套 `IF_HF.py` 使用的 HTML 可视化工具。**  
- `get_attention_html()`：橙色 heatmap，渲染 attention 权重
- `get_colored_html_from_ids()`：红/蓝双色，渲染 loss diff（红=loss增加/有害，蓝=loss减小/有益）
- `get_query_html_with_highlight()`：高亮目标 token 的 Query 展示
- `save_query_report_html()`：完整 EIF 分析 HTML 报告
- 目前 `tools/correlation-report/`（React 可视化工具）已承担可视化职责

---

### 🟡 有用但已退出主流程（Ground Truth 评估体系）

#### `construct_ground_truth.py`
**用 GPT API 生成 IF 算法的 Ground Truth 验证数据。**  
- 针对一个 test sample，生成 6 种变体（3 难度 × 正/负）：
  - **Level 1（浅层扰动）**：改变量名/注释等表面特征
  - **Level 2（深层逻辑改写）**：同一抽象逻辑，换场景重写
  - **Level 3（解耦知识点）**：只教一个核心子技能
- 输出为 `ground_truth_demo.jsonl`
- 设计意图：positive 变体应与 test 有正相关 IF 分数，negative 变体应有负相关分数

#### `evaluate_ground_truth.py`
**将生成的 GT 样本混入训练集，评估 IF 算法的检索准确率。**  
- 加载 100 个真实训练样本 + 6 个 GT 变体，混合成 106 样本训练集
- 运行 `NewInferenceFunction.influence_gradient_single()` 计算每个样本对 test 的 influence 分数
- 输出 GT 变体在全部样本中的排名（Rank / Percentile）
- 验证假设：正向变体应排在前面，负向变体应排在后面

#### `results_analysis.py`
**老版 EIF 结果的离线分析脚本。**  
- 读取 `experiment_results_v3.jsonl`（旧版批量实验产物）
- 绘制三张图：Self-Rank CDF（越陡越好）、Score 分布直方图、不同阈值下的 Success Rate 柱状图
- 同时打印统计摘要和需要重点审查的样本列表
- 该格式与当前 `correlation_matching_results.json` 不兼容，需适配后才能复用

---

#### 旧版文件（已在本目录存在）

| 文件 | 说明 |
|---|---|
| `NIF.py`（本目录中） | 更早期的 NIF 版本，早于上面的 `NIF_old.py` |
| `benchmark.py` | EIF 批量 benchmark 脚本，与旧版 NIF 配套 |
| `precompute.py` | 预计算训练集梯度的脚本，用于加速老版 IF 查询 |
