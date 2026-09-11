# ver7.1 代码定义核查报告

核查日期：2026-09-09  
核查对象：`ver7.1_code_definition_audit_checklist.md` 所列 P0、P1、P2 项目  
核查性质：代码、配置与结果文件的静态交叉审计；未将清单中的建议性文字视为算法事实。

## 1. 总体结论

本次核查得到四项关键结论。

1. **指标公式可以定稿。** Global、Opening-band 和 Balanced 均采用“两个输出通道分别求 standardized RMSE，再取算术平均”的实现，不是先合并通道 MSE 再开根号。
2. **文稿不得混用三套 refinement 协议。** 旧实验使用 8-case closure 和 top-\(r\) active set；20 参数实验使用完整输出头空间和 Gram 二次目标；P5/P6 使用 137 个训练工况、完整参数子空间和 matrix-free HVP。三者的目标函数、停止准则和 validation 规则不同。
3. **P5/P6 方法必须称为 inexact MF-L3CR。** 三次或五次 HVP 截断后，代码虽计算内层残差，但不要求残差达标才接受外层步。固定三步实验与最新 \(K=5\) 实验的 accepted steps 均未达到内层阈值。
4. **存在两项无法由现有材料确认的定义。** 当前工程中没有生成正式 `64\times64\times4` 训练张量的原始插值代码，也没有足以确认 MPa/mm 的单位元数据。论文不得把后处理 round-trip audit 当成训练数据生成方法，不得未经单位来源证明直接将乘回标准差后的误差标为 MPa 或 mm。

### 1.1 审计状态

| 状态 | 含义 |
|---|---|
| PASS | 代码、配置和结果相互一致，可据此定稿 |
| PARTIAL | 实现可确认，但论文表述必须限定适用实验或修改术语 |
| FAIL | 当前实现不支持清单或理论所暗示的表述 |
| UNRESOLVED | 缺少源代码或元数据，不能据现有文件下结论 |

### 1.2 三套不得混写的协议

| 协议 | 可训练空间 | 训练目标 | validation 规则 | 主要证据 |
|---|---:|---|---|---|
| AS-L3CR refinement | top-\(r\), \(r\in\{32,64\}\) | 固定 8 个训练案例的 closure | trial acceptance 强制不超过初始 validation 的 1.005 倍；预算内再选 validation 最优状态 | `test082301/run_high_accuracy_refinement.py` |
| Full-space 20-head L3CR | 20 | 137 个训练案例形成的精确 Gram 二次目标 | 高精度训练审计，不用 validation-best 回退决定优化终点 | `test082401/run_short_high_accuracy_experiment.py` |
| P5/P6 MF-L3CR | 44,149 / 86,653 | 137 个训练案例的确定性 full-data MSE | 训练步仅按训练目标和 ratio 接受；validation 只保存/选择预测评价状态 | `test090702/run_l3cr_inner_outer_experiment.py` |

论文若以 P6 为主实验，应以第三行为唯一 practical algorithm 定义。active-set、8-case closure 和 \(\varepsilon_{\rm act}\) 不能写入 P6 算法流程。

## 2. P0 核查结果

### P0-1. Global、Opening-band 与 Balanced

**状态：PASS。**

令 \(e_{n,c,q}=\widehat y_{n,c,q}-y_{n,c,q}\) 为标准化空间中的误差；\(c\in\{S_1,U\}\)；\(M_{n,q}\) 为材料掩码；\(H_{n,q}\) 为 opening-band 掩码。代码实际计算

\[
R_c^{\rm G}
=
\left(
\frac{\sum_{n,q}M_{n,q}e_{n,c,q}^2}
{\sum_{n,q}M_{n,q}}
\right)^{1/2},
\]

\[
R_c^{\rm O}
=
\left(
\frac{\sum_{n,q}H_{n,q}e_{n,c,q}^2}
{\sum_{n,q}H_{n,q}}
\right)^{1/2}.
\]

随后

\[
G=\frac{R_{S_1}^{\rm G}+R_U^{\rm G}}{2},
\qquad
O=\frac{R_{S_1}^{\rm O}+R_U^{\rm O}}{2},
\qquad
B=\frac{G+O}{2}.
\]

正式 seed-level 指标先汇集测试集全部有效体素再求 RMSE。Opening-band 只由 3 个具有非空 opening mask 的测试案例贡献，不是 19 个 case-level RMSE 的等权平均。

代码证据：

- `test080401/run_leakage_free_architecture_experiment.py:158-175`
- `test080401/run_leakage_free_architecture_experiment.py:176-184`

术语修正：变量名中的 `NRMSE` 并未除以 range、mean 或独立 reference norm。它是目标经过 z-score 后的 RMSE。正文宜写 **standardized RMSE**，并定义为上述公式。

### P0-2. Opening-band mask

**状态：PASS。**

对每个案例及每个厚度层分别执行：

1. 由原始 channel 3 构造材料集合 \(M=\{q:x_{q,3}>0\}\)；
2. 对 \(\neg M\) 做二维连通域标记；
3. 删除与平面外边界相连的 void components；
4. 剩余连通域定义为 internal openings；
5. 使用 \(5\times5\) 全 1 structuring element 膨胀一次；
6. 与材料掩码相交，排除孔洞内部和材料外部。

因此默认 opening band 是每个 \(z\) 层内的二维 \(L_\infty\) 网格半径 2 邻域：

\[
H=\operatorname{dilate}_{5\times5}(V_{\rm internal})\cap M.
\]

它不是 Euclidean distance threshold，也不在 \(z\) 方向膨胀。

代码证据：`test080401/run_leakage_free_architecture_experiment.py:47-72`。

### P0-3. Acceptance ratio

**状态：PASS。**

P5/P6 practical run 采用 cubic-model decrease：

\[
m_k(s)=g_k^Ts+\frac12s^TH_ks+\frac{\sigma_k}{6}\sum_i|s_i|^3,
\]

\[
\rho_k=
\frac{F(\theta_k)-F(\theta_k+s_k)}{-m_k(s_k)}.
\]

若 outer scaling 使用 \(s_k=\alpha_k\widetilde s_k\)，分母会按实际缩放步重新计算，而不是沿用 raw step 的预测下降。接受条件为

\[
-m_k(s_k)>0,
\qquad
F(\theta_k+s_k)<F(\theta_k),
\qquad
\rho_k\ge0.10.
\]

代码证据：

- local model：`test090702/run_progressive_full_parameter_l3cr.py:234-240`
- scaled ratio：`test090702/run_l3cr_inner_outer_experiment.py:478-502`

### P0-4. Validation safeguard

**状态：PARTIAL。不同实验不能共用一个公式。**

旧 AS-L3CR 实验使用 hard safeguard：

\[
F_{\rm val}(\theta_{k+1})
\le
1.005F_{\rm val}(\theta_0).
\]

该不等式是 active-subspace trial 的 acceptance condition。证据：`test082301/run_high_accuracy_refinement.py:646-649,681-699`。

P5/P6 MF-L3CR **没有**上述 validation acceptance constraint。其外层接受只取决于训练下降和 \(\rho_k\)。每个已接受状态计算 validation objective；只有 validation 改善时才保存 `validation_best.pt`。最终测试评价比较初始 checkpoint 与 refinement 过程中的 validation-best，并取 validation 更低者。证据：

- 保存 validation-best：`test090702/run_l3cr_inner_outer_experiment.py:519-525`
- 评价时与初始状态比较：`test090702/run_l3cr_inner_outer_experiment.py:775-795`

论文若描述 P5/P6，应写成“validation-based model selection”，不能写成 trial-level safeguard inequality。

### P0-5. Active-subspace stopping

**状态：PARTIAL。只适用于旧 AS-L3CR。**

旧实验定义

\[
r_{\rm act}(\theta)
=
\frac{\|\operatorname{Top}_r(|\nabla F(\theta)|)\|_2}
{\max\{1,\|\operatorname{Top}_r(|\nabla F(\theta_0)|)\|_2\}},
\]

并以 \(r_{\rm act}\le10^{-6}\) 作为提前停止条件。证据：`test082301/run_high_accuracy_refinement.py:309-316,724-733`。

注意：分子在每次记录时重新对当前梯度取 top-\(r\)，不一定是当前 L3CR step 使用的同一个 active index set。因此它是“当前 top-\(r\) 梯度范数”，不是固定子空间上的 stationarity certificate。

P5/P6 为 full selected-module space，不存在 \(\varepsilon_{\rm act}\)。当前 \(K=5\) 正式实验按目标 accepted-step 数停止，不按梯度阈值停止。

### P0-6. Top-\(r\) active set

**状态：PASS，限旧 AS-L3CR。**

- 对所有 trainable parameter tensors 按 `model.parameters()` 顺序展平；
- 对 \(|g_i|\) 调用 `torch.topk(..., sorted=True)`；
- 不做 parameter-tensor normalization 或 layer normalization；
- 每个 outer iteration 重新计算 active set；
- 一个 inner solve 内 active set 固定；
- rejected outer trial 后下一轮重新计算。由于参数被恢复，除数值 tie 外 active set 通常相同；
- tie 的选择顺序未由代码显式规定，不能宣称 deterministic tie-breaking。

证据：`test082301/run_high_accuracy_refinement.py:598-616,661-670`。

### P0-7. Practical step 与理论 admissibility

**状态：FAIL，若文稿把 practical run 直接纳入定理保证。**

P5/P6 practical solver 的事实为：

\[
r_m(s)=
\frac{\left\|g+Hs+\frac{\sigma}{2}|s|\odot s\right\|_{3/2}}
{\max\{1,\|g\|_{3/2}\}}.
\]

- 固定三步实验：inner HVP cap \(K=3\)，nominal tolerance \(10^{-2}\)；
- 最新正式实验：inner HVP cap \(K=5\)，nominal tolerance 仍为 \(10^{-2}\)；
- 第一个 HVP 用于 Rayleigh step-size estimate；其余 HVP 用于 PG trial 或 inner backtracking；
- residual 被计算并记录，但 **不作为 outer acceptance 的必要条件**；
- 未实现独立的 Cauchy-decrease certificate；
- 实际检查的 model condition 只有 \(-m_k(s_k)>0\)；
- 固定三步和 \(K=5\) 正式结果中的 accepted steps 均标记为 `INEXACT_DESCENT`。

证据：`test090702/run_l3cr_inner_outer_experiment.py:353-393,473-526`，以及 `test090702/k5_l3cr_step_trace.csv`。

论文必须采用如下边界：

> The convergence statement applies to steps satisfying the stated full-space admissibility conditions. The P5/P6 realization truncates the inner proximal-gradient iteration after a prescribed HVP budget and accepts an inexact step only when it yields positive model reduction, strict full-data objective decrease, and \(\rho_k\ge0.10\). Hence, the practical trajectory is reported as an empirical inexact MF-L3CR realization and is not claimed to satisfy the theorem's inner-residual condition.

### P0-8. Inner step parameter 与 \(\sigma_k\)

**状态：PARTIAL。论文中的 \(\beta_{k,t}\) 与代码变量不一致。**

代码没有 `beta`；它直接更新 proximal step size \(\alpha_t\)：

\[
\alpha_0=
\frac{0.8}
{\max\left\{
|g^THg|/\|g\|_2^2,10^{-6}
\right\}}.
\]

若 quadratic majorization 通过，\(\alpha\leftarrow1.15\alpha\)；失败时 \(\alpha\leftarrow0.5\alpha\)。若论文坚持用 \(\beta=1/\alpha\)，则相应为 successful inner trial 后 \(\beta\leftarrow\beta/1.15\)，failure 后 \(\beta\leftarrow2\beta\)。代码没有独立的 \(\beta_{\min},\beta_{\max}\)。

P5/P6 使用

\[
\sigma_0=10^{-2},
\qquad
\sigma_{k+1}=
\begin{cases}
\max\{10^{-12},0.5\sigma_k\},&\text{accepted and }\rho_k\ge0.75,\\
\sigma_k,&\text{accepted and }0.10\le\rho_k<0.75,\\
\min\{10^{12},2\sigma_k\},&\text{rejected}.
\end{cases}
\]

Outer scaling 最多测试 5 个尺度 \(1,1/2,1/4,1/8,1/16\)。证据：`test090702/run_l3cr_inner_outer_experiment.py:353-390,485-512,536-539`。

## 3. P1 核查结果

### P1-1. KAN block

**状态：PASS。**

正式 leakage-free BAM-KAN 使用 Gaussian-basis KAN，不是 spline KAN。对输入 \(x\in\mathbb R^d\)，单层第 \(o\) 个输出为

\[
\mathcal K_o(x)
=
\sum_{i=1}^{d}W_{oi}x_i+b_o^{\rm lin}
+
\sum_{i=1}^{d}\sum_{j=1}^{5}
c_{oij}
\exp\!\left[-\left(\frac{x_i-\mu_j}{w_i}\right)^2\right]
+b_o,
\]

其中 \(\mu_j\in\{-2,-1,0,1,2\}\) 固定，\(w_i=\operatorname{clip}(e^{\ell_i},0.20,2.50)\) 可训练，初值为 0.75；\(c_{oij}\) 可训练；同时保留 full linear path 和额外 bias。

Residual KAN block 为

\[
x^+=x+a\tanh\!\left(\mathcal K(\operatorname{LN}(x))\right),
\qquad a_0=0.10,
\]

其中 \(a\) 是可训练标量。fine 与 coarse KAN 各串联两个 block；正式宽度均为 12。KAN 在每个 pooled voxel 上独立执行，不在 KAN layer 内进行空间卷积或邻域 mixing。

证据：`test071407/run_bampikan_unet_cnn_comparison.py:103-129,187-220`。

### P1-2. Local pathway

**状态：PASS。**

正式 BAM-KAN 的 `spatial_width=19`。8 个物理输入经 \((1,3,3)\) stem 映射为 15 通道；3 个坐标输入经 \((1,1,1)\) stem 映射为 4 通道。拼接后依次通过两个 19 通道 anisotropic residual blocks，kernel 均为 \((1,3,3)\)，平面 dilation 分别为 1 和 2。

因此 local pathway 真实存在。它在该阶段不交换相邻厚度层信息；厚度耦合由 coarse projection、coarse spatial block 和 \(U\)-head 中的 \((3,3,3)\) convolution 提供。

### P1-3. Local、fine、coarse 与 heads

**状态：PASS。**

输入尺寸按 \((N,C,D,H,W)=(N,11,4,64,64)\) 记：

- local：8-channel physical stem 与 3-coordinate stem 分离编码，输出 \(19\times4\times64\times64\)；
- fine：local 经 1x1x1 projection 得 12 通道，平面 average pooling 到 \(4\times32\times32\)，两层 residual KAN，再 trilinear 上采样；
- coarse spatial：local 平面 pooling 到 \(4\times32\times32\)，经 \((3,3,3)\) projection 得 28 通道，再经一个 28 通道 3D residual block；
- coarse KAN：28 通道投影到 12，再次平面 pooling 到 \(4\times16\times16\)，两层 residual KAN，直接上采样到 full resolution；
- fusion：\(19+12+12=43\) 通道经 1x1x1 block 回到 19；
- \(S_1\) head：fused 与 fine 拼接，经 \((1,3,3)\) block 和 1x1x1 output convolution；
- \(U\) head：fused 与 coarse-KAN 拼接，经 \((3,3,3)\) block 和 1x1x1 output convolution。

证据：`test071407/run_bampikan_unet_cnn_comparison.py:149-245`。

### P1-4. P5/P6 trainable modules

**状态：PASS。**

| Top-level module | 参数量 | P5 | P6 |
|---|---:|:---:|:---:|
| `physical_stem` | 1,125 | yes | yes |
| `coordinate_stem` | 24 | yes | yes |
| `local` | 13,224 | yes | yes |
| `coarse_projection` | 14,448 | yes | yes |
| `coarse_spatial` | 42,504 | **no** | yes |
| `fine_projection` | 264 | yes | yes |
| `fine_kan` | 1,850 | yes | yes |
| `coarse_kan_projection` | 372 | yes | yes |
| `coarse_kan` | 1,850 | yes | yes |
| `fusion` | 874 | yes | yes |
| `s1_head` | 2,548 | yes | yes |
| `u_head` | 7,570 | yes | yes |
| **合计** | **86,653** | **44,149** | **86,653** |

P5 不是“除 stems 外的网络”，而是 **除 `coarse_spatial` 外的全部参数**。P6 相对 P5 只新增 42,504 参数的 `coarse_spatial` block。证据：`test090702/run_progressive_full_parameter_l3cr.py:43-52,90-104`。

### P1-5. Full-data objective 与 micro-batch accumulation

**状态：PASS。**

137 个 optimization-training cases 共含 \(M=2,159,924\) 个材料体素。代码目标为

\[
F(\theta)=
\frac{1}{M}
\sum_{n,q}M_{n,q}
\left[
\frac{(\widehat S_{1,n,q}-S_{1,n,q})^2
+(\widehat U_{n,q}-U_{n,q})^2}{2}
\right].
\]

每个 micro-batch 返回 masked loss **sum**，所有 batch 累加后统一除以 \(M\)。梯度与 HVP 也先累加未归一化 batch contributions，再统一除以 \(M\)。最后不足 4 个案例的 batch 不会获得额外权重。

证据：`test090702/run_l3cr_inner_outer_experiment.py:150-227`。

### P1-6. FD relative error

**状态：PASS。**

随机单位方向 \(v\) 下，实际有限差分为

\[
d_h=
\frac{\nabla F(\theta+hv)-\nabla F(\theta-hv)}{2h},
\qquad
h=h_{\rm base}(1+\|\theta\|_2),
\]

\[
e_{\rm FD}=
\frac{\|Hv-d_h\|_2}
{\max\{\|Hv\|_2,\|d_h\|_2,10^{-12}\}}.
\]

审计使用 \(h_{\rm base}\in\{10^{-3},10^{-4},10^{-5}\}\)，表中报告三个尺度中的最小 relative error。证据：`test090702/run_l3cr_inner_outer_experiment.py:295-328`。

### P1-7. Hessian symmetry error

**状态：PASS。**

对独立单位方向 \(u,v\)，代码计算

\[
e_{\rm sym}=
\frac{|u^THv-v^THu|}
{\max\{|u^THv|,|v^THu|,10^{-12}\}}.
\]

这不是显式 Hessian 的 Frobenius symmetry error。证据：`test090702/run_l3cr_inner_outer_experiment.py:295-311`。

### P1-8. Micro-batch consistency

**状态：PASS，但术语应改。**

代码在 micro-batch \(1,2,4\) 下分别计算相同 full-data objective，并报告

\[
\delta_{\rm mb}=\max_bF_b-\min_bF_b.
\]

它是绝对 spread，不是 relative error，也没有对 gradient/HVP 做 micro-batch consistency 比较。P5/P6 实测均为 \(1.38778\times10^{-16}\)。证据：`test090702/run_l3cr_inner_outer_experiment.py:288-294,339-348`。

### P1-9. WBP / derivative-work counter

**状态：PARTIAL。**

20 参数审计定义 implementation-normalized work unit

\[
W_{\rm BP}=N_g+N_{Hv}.
\]

一次 objective-gradient 记 1；显式构造 \(n\times n\) Hessian 时，\(n\) 个 Hessian columns 记 \(n\) 个 HVP，再加 1 个 gradient；用于记录终点梯度的额外调用也计入。L-BFGS 每次 closure call 记 1。证据：`test082401/full_space_l3cr_core.py:170-185,216-244,247-303`。

该计数假定 1 HVP 与 1 gradient 等价，只适合算法工作量归一化。它不是实际反向传播次数，也不能代替 wall-clock 或实测 \(c_{Hv}=t_{Hv}/t_g\)。P5/P6 论文宜同时报告 \(N_g,N_{Hv}\) 与 wall-clock，不宜沿用“WBP 更快”的无条件表述。

### P1-10. Matched-budget comparator

**状态：PARTIAL。**

固定三步及 \(K=5\) 实验先记录每个 seed 的 L3CR accepted-step snapshot 实际累计时间，再把该时间作为 AdamW/AdamW-0/L-BFGS 的 allocation。基线在启动一次 update、gradient 或 validation 前，用历史 operation-time estimate 检查剩余预算；预计不能完整结束时提前停止。因此

\[
T_{\rm comparator}\le T_{\rm L3CR},
\]

但通常不严格相等。\(K=5\) 五 seed 平均时间分别为 2550.6 s（L3CR）、2493.6 s（Resumed AdamW）、2497.9 s（AdamW-0）和 2450.8 s（L-BFGS）。

论文宜写 **matched wall-clock allocation with no operation overshoot**，并同时列出实际消耗时间；不能只写 exact equal wall-clock。

### P1-11. L-BFGS 配置

**状态：PASS。**

P5/P6 当前实现使用 PyTorch L-BFGS：

```text
lr = 1.0
max_iter = 1 per optimizer.step call
max_eval = 20
history_size = 20
line_search_fn = strong_wolfe
tolerance_grad = 1e-14
tolerance_change = 1e-16
```

每次 line-search closure 调用均执行完整 137-case backward，并计入 gradient/objective counters。若一次 outer update 后训练目标未下降，路线以 `STOP_NO_DESCENT` 结束。证据：`test090702/run_l3cr_inner_outer_experiment.py:630-683`。

旧 AS-L3CR 对照使用不同 tolerance：`1e-9` 和 `1e-12`，`max_eval=10`。该配置不能与 P5/P6 配置混写。

### P1-12. Resumed AdamW 配置

**状态：PASS。**

P5/P6 当前实现加载 warm-up checkpoint 的 first/second moments。学习率设为

\[
\alpha_{\rm refine}=0.3\alpha_{\rm checkpoint}.
\]

`Resumed-AdamW` 使用 weight decay \(10^{-5}\)；`AdamW-0` 使用 0。两者恢复同一 optimizer moments，差别仅为 weight decay。refinement 不使用 scheduler，学习率固定。证据：`test090702/run_l3cr_inner_outer_experiment.py:590-632`。

### P1-13. 20-parameter output head

**状态：PASS。**

20 个参数为：

```text
s1_head.1.weight: 9
s1_head.1.bias:   1
u_head.1.weight:  9
u_head.1.bias:    1
```

两项 weight 均为 \((1,9,1,1,1)\) 的 1x1x1 convolution。证据：`test082401/run_short_high_accuracy_experiment.py:184-196`。

### P1-14. \(F^\ast\) 与 Gram-form loss

**状态：PASS。**

冻结主干后，每个输出通道形成带 bias 的 10 维 design vector。代码累计 \(X^TX\)、\(X^Ty\) 与 \(y^Ty\)，形成 20 维 block-diagonal quadratic：

\[
F(w)=\frac12w^THw-b^Tw+c.
\]

其中两个 10x10 Gram blocks 除以材料体素数；\(b\) 同样归一化。参考解由

\[
w^\ast=\operatorname{lstsq}(H,b;\operatorname{rcond}=10^{-12})
\]

获得，\(F^\ast=F(w^\ast)\)。代码同时审计 Gram 目标与直接完整训练损失的一致性和 Hessian rank。证据：`test082401/run_short_high_accuracy_experiment.py:212-263`。

### P1-15. Closure set

**状态：PARTIAL。只适用于旧 AS-L3CR。**

固定索引为

\[
\{7,21,29,63,84,89,95,158\}.
\]

它们均属于 137-case optimization-training subset，与 30-case validation set 不重合，其中 63 和 84 为 opening-mask 非空案例。现有代码没有记录从候选训练案例生成该集合的算法或随机种子。因此只能称为 **prespecified fixed closure containing two opening cases**，不能声称随机抽样、分层抽样或代表性最优。

P5/P6 不使用该 closure；它使用全部 137 个 optimization-training cases。

### P1-16. Held-out validation

**状态：PASS，需区分协议。**

固定 split seed 为 20260629。167-case development pool 被划分为 137 train 与 30 validation；五个训练 seed 共享同一划分。旧 AS-L3CR 中 validation 参与 trial safeguard 和预算内 checkpoint selection；P5/P6 中 validation 不参与 step acceptance，只用于保存 validation-best 和最终预测状态选择。19-case test set 不参与上述决策。

### P1-17. 随机种子

**状态：PARTIAL。**

网络 warm-up 的 seed 调用 `random.seed`、`numpy.random.seed` 和 `torch.manual_seed`，并控制模型初始化与 epoch 内 training-case permutation。train/validation split 由独立固定 seed 20260629 控制。P5/P6 refinement 本身为固定 full-data 顺序；HVP audit 的随机方向由 \(20260826+n_{\rm param}\) 控制。

代码没有调用 `torch.use_deterministic_algorithms(True)`。因此可以写“seeded and deterministic under the recorded CPU execution path”，不宜声称跨平台 bitwise reproducibility。

### P1-18. Normalization

**状态：PARTIAL，存在容易误写的范围问题。**

对保留输入通道和两个目标通道，代码按 raw `train.npy` 中 **全部 167 个 development cases** 的材料体素计算 population mean 和 population standard deviation：

\[
\mu_c=\frac1{N_c}\sum_{(n,q)\in M}x_{n,q,c},
\qquad
s_c=\left[\frac1{N_c}\sum_{(n,q)\in M}(x_{n,q,c}-\mu_c)^2\right]^{1/2},
\]

\[
\widetilde x=(x-\mu)/s,
\qquad
\widetilde y=(y-\mu_y)/s_y.
\]

若 \(s\le10^{-12}\)，代码将其替换为 1。坐标通道由各轴等距映射到 \([-1,1]\)，不再用上述均值和标准差归一化。

重要限制：normalization 在固定 137/30 split 之前计算，因而包含 30 个 validation cases；它不包含 19 个 test cases。论文不能写“statistics were fitted on the 137-case optimization-training subset”。准确写法是“fitted on the 167-case development pool, with the test set excluded”。若目标是完全无 validation preprocessing leakage，需要重新按 137 cases 计算统计量并重训所有模型。

证据：`test080401/run_non_abaqus_plan_audit.py:187-231`；`test080401/run_leakage_free_architecture_experiment.py:87-104`。

### P1-19. Common-grid interpolation

**状态：UNRESOLVED。**

当前工程可确认的是一个独立 round-trip audit：每个 Excel case 内坐标先按 case 范围归一化；`scipy.interpolate.griddata` 先做 linear interpolation，convex hull 外用 nearest fill，再由 grid 线性回插节点并对缺失项 nearest fill。

证据：`test080401/audit_excel_projection.py:103-123`。

但是现有目录中没有生成正式 `train.npy/test.npy` 或 `train_inputs.npy/test_inputs.npy` 的原始 FE-to-grid 程序。因此不能证明正式训练张量就是按上述算法构造。论文应将 round-trip audit 明确写成 post hoc projection audit；正式数据生成方法仍需原始脚本或导出记录支持。

### P1-20. Distance-to-opening channel

**状态：FAIL，若正文声称模型输入含该通道。**

正式 leakage-free 输入为 raw channels \([2,3,5,6,7,8,9,11]\) 加 \(x,y,z\) 坐标。现有 provenance 表只把 retained channels 统称为 geometry/material/control fields，没有任何代码计算 Euclidean、Manhattan 或 grid distance-to-opening。Opening mask 仅用于评价，不作为 BAM-KAN 输入。

因此应删除“distance-to-opening input/prior”表述。除非补充原始 channel map，否则也不能把 raw channel 2 命名为 opening distance。

## 4. P2 核查结果

### P2-1. 参数量

**状态：PASS。**

统一计数函数为

```python
sum(p.numel() for p in model.parameters() if p.requires_grad)
```

正式 leakage-free 模型参数量为：CNN 85,450；U-Net 86,690；BAM-KAN 86,653。对应配置为 CNN width 22、U-Net width 14、BAM-KAN spatial/fine/coarse widths \(19/12/12\)。证据：`test080401/formal_leakage_free/model_specifications.json`。

旧 `test071407/formal` 使用另一输入和网络配置，参数量为 87,232/87,824/87,838。两组数值不得在同一正式表中混用。

### P2-2. Opening-band radius sensitivity

**状态：PASS。**

radius \(r\in\{1,2,3,4\}\) 只改变二维 dilation kernel 为 \((2r+1)\times(2r+1)\)。模型 checkpoint、test cases、prediction、normalization 和 Global mask 均固定，没有重新训练或重新选择 checkpoint。证据：`test080401/compute_hole_radius_sensitivity.py:39-98`。

可直接写：

> Only the evaluation-band radius is varied; model weights, predictions, test cases, normalization statistics, and all remaining evaluation settings are fixed.

### P2-3. Physical-scale RMSE

**状态：PARTIAL。**

代码采用

\[
\operatorname{RMSE}^{\rm raw}_c=s_{y,c}\operatorname{RMSE}^{\rm std}_c.
\]

对 z-score normalization，该式与先反标准化 prediction/target 再计算 RMSE 等价，均值 \(\mu_y\) 在误差中抵消。证据：`test080401/run_leakage_free_architecture_experiment.py:168-175`。

但当前数据卡未给出 raw channels 0/1 的单位。代码也没有执行 Pa-to-MPa 或 m-to-mm 换算。因此目前只能称为 **raw-scale RMSE**。标注 MPa/mm 前必须补充 Abaqus 导出单位制，并在表中明确换算因子。

### P2-4. Table 6 的 30 个 accepted steps

**状态：PASS，限固定三步 P5/P6 实验。**

| Stage | outer trials | accepted | rejected | min. accepted \(\rho\) | inner majorization backtracks |
|---|---:|---:|---:|---:|---:|
| P5 | 15 | 15 | 0 | 0.9642737 | 4 |
| P6 | 15 | 15 | 0 | 0.9613974 | 4 |

总数为 \(5\text{ seeds}\times2\text{ stages}\times3\text{ accepted steps}=30\)。所有 outer trials 均接受，故“all-trial minimum rho”与“accepted-trial minimum rho”相同，为 0.9613974。这里的 4+4 是 inner quadratic-majorization backtracks，不是 rejected outer trials。

证据：`test090401/l3cr_stepwise_descent.csv`、`test090401/optimization_trace.csv`、`test090401/validation_results.csv`。

### P2-5. Hardware 与计时

**状态：PARTIAL。**

当前机器为 Apple M4 Mac mini，10 cores，32 GB RAM，macOS 15.5。正式 P5/P6 运行使用：

```text
Python 3.12.2
PyTorch 2.5.1
NumPy 1.26.4
CPU, float64
torch intra-op threads = 1
torch inter-op threads = 1
BLAS/OpenMP threads = 1
micro-batch = 4
timer = time.perf_counter
```

当前 Conda/PyTorch build 报告 `x86_64`，而主机为 Apple Silicon；MPS unavailable。这表明实验运行于 x86_64 用户态兼容路径，可能包含 Rosetta 转译成本。硬件段必须披露这一点，否则 wall-clock 不具备充分可比性。

P5/P6 CPU 操作无需 CUDA/MPS synchronization。HVP audit 为单次方向测量，没有独立 warm-up repetitions；因此表中的 full-gradient/full-HVP 秒数是该环境下的 operation measurement，不应表述为多次 benchmark 均值。

## 5. 二十项明确答案

1. Global：两个输出通道在材料体素上的 standardized RMSE 之平均。
2. Opening：两个输出通道在二维 radius-2 opening band 上的 pooled standardized RMSE 之平均。
3. Balanced：\((G+O)/2\)。
4. Opening mask：逐厚度层删除 border-connected void，内部 void 用 5x5 kernel 膨胀一次并与材料相交。
5. Practical \(\rho\)：full-data actual decrease 除以实际缩放步的 cubic-model predicted decrease。
6. Validation：旧 AS-L3CR 用 1.005 hard limit；P5/P6 不用于 step acceptance，只做 validation-best selection。
7. Active residual：当前 top-\(r\) 梯度的 2-norm 除以初始 top-\(r\) norm，阈值 \(10^{-6}\)；仅适用于旧 AS-L3CR。
8. Active set：flattened absolute gradient top-\(r\)，每个 outer iteration 更新，inner iteration 固定，无 layer-wise normalization。
9. \(\sigma\)：初值 \(10^{-2}\)，very successful 减半，accepted-middle 不变，rejected 加倍；代码使用 \(\alpha\) 而非 \(\beta\)。
10. Inner residual：normalized \(\ell_{3/2}\) norm of \(g+Hs+(\sigma/2)|s|\odot s\)。
11. HVP cap：\(K=3\) 或最新 \(K=5\)；residual 会记录但不限制 acceptance，故为 inexact realization。
12. KAN：5 个固定 Gaussian centers、trainable per-input widths、trainable coefficients、linear path、bias、tanh block activation。
13. KAN block：LayerNorm、AdaptiveKAN、trainable residual scale；fine/coarse 各两层，width 12。
14. local/fine/coarse：19-channel anisotropic local path、12-channel half-resolution fine KAN、12-channel quarter-resolution coarse KAN，并含 28-channel 3D coarse spatial path。
15. P5/P6：P5 冻结 `coarse_spatial`，其余模块可训练；P6 全部 86,653 参数可训练。
16. Full-data aggregation：batch loss sums 按材料体素总数统一归一化；两通道先等权平均。
17. FD/symmetry：分别采用 central gradient difference relative error 与 bilinear-form symmetry relative error。
18. WBP：\(N_g+N_{Hv}\) 的 nominal work unit；不是 wall-clock，也不是严格硬件成本。
19. Matched budget：L3CR snapshot 时间作为 allocation，comparator 采用 operation guard 防止超时；实际消耗通常略小。
20. Closure/validation：8-case closure 只用于旧 AS-L3CR；P5/P6 使用 137-case full-data objective；30-case validation 固定且 test set 不参与选择。

## 6. 必须修改的论文表述

### 6.1 必改

1. 将 P5/P6 方法统一命名为 **inexact matrix-free \(\ell_3\)-regularized refinement** 或 **inexact MF-L3CR**。
2. 删除“P5/P6 practical steps satisfy the theorem's admissibility conditions”。改为理论参考方法与截断实现分开陈述。
3. P6 算法段删除 top-\(r\)、active-gradient stopping 和 8-case closure；这些只能出现在独立的旧实验说明中。
4. 将 P5/P6 validation safeguard 改成 validation-based state selection，不写 1.005 trial constraint。
5. 将 `NRMSE` 解释改为 standardized RMSE；不要暗示 range normalization。
6. 删除未经证实的 distance-to-opening input。
7. 将 normalization 范围改为 167-case development pool，不写 137-case training subset。
8. 在单位来源补齐前，将 MPa/mm 改为 raw-scale units 或删除单位标签。
9. 将 `micro-batch consistency error` 改为 `absolute full-objective spread across micro-batch sizes 1, 2, and 4`。
10. 将 `equal wall-clock` 改为 `matched wall-clock allocation`，并列出各方法实际运行时间。

### 6.2 建议补充

1. 给出正式 FE-to-grid 数据生成脚本、插值策略、坐标范围、外推和缺失值处理；否则保留为数据来源限制。
2. 给出 Abaqus 一致单位制及 raw channel 0/1 的单位转换表。
3. 保存 retained raw channels 2/3/5/6/7/8/9/11 的语义、单位和生成来源；当前“geometry/material/control field”不足以复现。
4. 披露 x86_64 PyTorch 在 Apple M4 主机上的兼容执行路径，避免将绝对秒数外推到原生 ARM、CUDA 或 MPS。
5. 对最新 \(K=5\) 结果明确报告：25/25 steps 严格降低训练目标，但 0/25 满足 nominal inner residual tolerance；不能据此宣称高精度内层求解。

## 7. 可直接采用的算法定位

建议正文将方法定位为：

> We use deterministic full-data Hessian-vector products to construct an inexact matrix-free refinement of the \(\ell_3\)-regularized local model. The inner proximal-gradient iteration is truncated by a prescribed HVP budget. A candidate is accepted only if the scaled local model predicts a positive reduction, the complete 137-case training objective decreases strictly, and the actual-to-predicted reduction ratio is at least 0.10. The inner residual is reported diagnostically; it is not imposed as an acceptance certificate in the P5/P6 experiments. Accordingly, the theoretical statement for admissible full-space steps is not asserted for the truncated practical trajectory.

这一定位与现有代码和日志一致。它支持“全参数 matrix-free HVP 可执行”和“有限步训练下降”，不支持“practical P6 已满足理论高精度子问题条件”或“已证明测试泛化优势”。

## 8. 证据索引

| 内容 | 证据文件 |
|---|---|
| 指标、mask、normalization load | `test080401/run_leakage_free_architecture_experiment.py` |
| 数据筛选与 normalization fit | `test080401/run_non_abaqus_plan_audit.py` |
| BAM-KAN/KAN architecture | `test071407/run_bampikan_unet_cnn_comparison.py` |
| Active-subspace、closure、旧 validation safeguard | `test082301/run_high_accuracy_refinement.py` |
| 20 参数 Gram problem 与 \(F^\ast\) | `test082401/run_short_high_accuracy_experiment.py` |
| 20 参数 WBP 与 full-space solver | `test082401/full_space_l3cr_core.py` |
| P5/P6 stage definitions | `test090702/run_progressive_full_parameter_l3cr.py` |
| 最新 full-data MF-L3CR 和 baselines | `test090702/run_l3cr_inner_outer_experiment.py` |
| 固定三步 P5/P6 审计 | `test090401/hvp_audit_summary.csv`, `test090401/validation_results.csv` |
| 最新 \(K=5\) 结果 | `test090702/k5_per_seed_metrics.csv`, `test090702/k5_l3cr_step_trace.csv` |
| radius sensitivity | `test080401/compute_hole_radius_sensitivity.py`, `test080401/13_hole_radius_metadata.json` |
| projection round-trip audit | `test080401/audit_excel_projection.py`, `test080401/14_projection_audit.csv` |

## 9. 投稿就绪判定

当前代码定义审计：**通过，但论文不能原样投稿。**

下列条件完成后，算法与实验定义可达到可复现稿件要求：

- [ ] 三套 refinement 协议在正文和附录中完全分离；
- [ ] P5/P6 全部改称 inexact MF-L3CR；
- [ ] 指标改为 standardized RMSE 并给出 pooled-mask 公式；
- [ ] 删除 distance-to-opening input 主张；
- [ ] 修正 normalization fit 范围；
- [ ] 补充原始 FE-to-grid 生成代码，或明确列为不可复现限制；
- [ ] 补充单位元数据后再使用 MPa/mm；
- [ ] wall-clock 表披露 Apple M4、32 GB、x86_64 PyTorch compatibility path、CPU single-thread、float64；
- [ ] 理论定理只对应满足 admissibility conditions 的 reference method，不覆盖截断 practical trajectory；
- [ ] 最新 \(K=5\) 结果同时报告训练目标下降与内层残差未达标，不选择性呈现。
