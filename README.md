# Spectral-peaks

**纯 Python + NumPy 从零实现的光谱寻峰工具包 —— 不需要 SciPy。**

一个轻量包，内含基线校正、Savitzky–Golay 平滑、三套彼此独立的寻峰算法、峰指标计算，
以及 Levenberg–Marquardt 峰形拟合。SciPy 只被测试套件用来做**交叉验证**，未安装 SciPy
时这些测试会自动跳过。

```python
import numpy as np
from spectral_peaks import detect_peaks

# 合成光谱：指数背景 + 4 个高斯峰 + 噪声
x = np.linspace(400.0, 2400.0, 2001)
y = 1.5 * np.exp(-(x - x[0]) / 700.0)
for centre, height, fwhm in [(520, 1.0, 24), (905, 1.6, 22), (1300, 0.9, 45), (1700, 1.2, 36)]:
    y += height * np.exp(-0.5 * ((x - centre) / (fwhm / 2.3548)) ** 2)
y += np.random.default_rng(3).normal(scale=0.005, size=x.size)

result = detect_peaks(y, x=x, baseline="airpls", smooth=11, max_width=120)

print(result.table(digits=4))
# index     x  height  prominence  width   area    snr
# -----  ----  ------  ----------  -----  -----  -----
#   120   520  0.9953       0.995  24.12  25.33  164.4
#   505   905   1.596       1.598  22.05  37.22  263.6
#   900  1300  0.8638      0.8723   43.8  40.02  142.7
#  1300  1700   1.179       1.187  35.88  44.55  194.7

result.positions()          # array([ 520.,  905., 1300., 1700.])
result.to_csv("peaks.csv")  # 导出完整表格
```

真值为 520/905/1300/1700、半高全宽 24/22/45/36：峰位完全准确，峰宽误差小于 3 %，
面积误差小于 4 %。

## 功能一览

| 模块 | 提供的内容 |
| --- | --- |
| **平滑** | Savitzky–Golay 滤波与系数（支持任意阶导数）、滑动平均、高斯平滑、5 种边缘填充方式 |
| **基线** | airPLS、非对称最小二乘（ALS）、SNIP、形态学开运算、滚球法、迭代多项式，以及统一的 `correct_baseline()` 入口 |
| **寻峰** | `find_peaks()`（与 SciPy 兼容的局部极值 + 属性滤波）、`find_peaks_derivative()`（一/二阶导数法）、`find_peaks_cwt()`（Ricker 小波脊线检测） |
| **指标** | 稳健噪声估计、SNR、地形学 prominence、任意相对高度处的峰宽/半高全宽、峰面积 |
| **拟合** | 高斯 / 洛伦兹 / 伪 Voigt / Voigt 峰形，自实现的 Levenberg–Marquardt（含参数误差与箱式约束）、多峰共用背景的同时拟合 |
| **流程** | `detect_peaks()` 串联「基线 → 平滑 → 寻峰 → 指标 → 可选拟合」，返回 `Peak` 对象列表 |

所有代码只依赖 NumPy 数组。连基线平滑所需的线性代数（带状 LDLᵀ 分解）都是自己实现的，
因此包本身零额外依赖，同时对数万点的长光谱依然足够快（2 万点的 airPLS 单步约 0.1 s）。

## 安装

包位于 `src/`，可以直接安装，也可以把它加入 `sys.path`：

```bash
pip install -e .            # 运行依赖：numpy
pip install -e ".[test]"    # 额外安装 pytest 与 scipy（用于测试）
```

```python
import spectral_peaks
print(spectral_peaks.__version__)   # '0.1.0'
```

## 快速开始

### 1. 检测峰

```python
from spectral_peaks import detect_peaks

result = detect_peaks(
    y, x=x,
    method="prominence",        # "prominence" | "derivative" | "cwt"
    baseline="airpls",          # BASELINE_METHODS 中的任意名字，或直接给基线数组
    baseline_kwargs={"lam": 1e5},
    smooth=11,                  # Savitzky-Golay 窗口长度（样点）
    prominence=0.05,            # 滤波条件，与 scipy.signal.find_peaks 一致
    distance=5,
    fit_model="gaussian",       # 可选：对每个峰做局部峰形拟合精修
)

for peak in result:             # PeakResult 可迭代、可索引、可取长度
    print(peak.x, peak.height, peak.fwhm, peak.area, peak.snr)
```

每个 `Peak` 带有 `index`、`x`、`height`、`prominence`、`left_base`、`right_base`、
`width`（`rel_height` 处的峰宽）、`left_ip`、`right_ip`、`area`、`snr`，以及可选的 `fit`
字典。只要传入了 `x` 轴，峰宽与面积就以 **x 轴单位**给出，否则以样点为单位。

当没有显式给出 `height`、`prominence` 或 `threshold` 时，`detect_peaks` 会施加一个默认的
5σ prominence 门限（基于处理后信号的稳健 MAD 尺度），把噪声涟漪和基线校正残留挡在结果之外。

### 2. 基线校正

```python
from spectral_peaks import correct_baseline

corrected, baseline = correct_baseline(y, method="airpls", return_baseline=True)
snip = correct_baseline(y, method="snip", iterations=40)
ball = correct_baseline(y, method="rolling_ball", radius=40)
```

可选方法：`airpls`、`als`、`snip`、`morphological`、`rolling_ball`、`polynomial`。

### 3. 峰形拟合（含重叠峰分解）

```python
from spectral_peaks import fit_peak, fit_peaks, fit_region

# 在局部窗口内拟合单峰（同时拟合线性背景）
fit = fit_peak(x, y, center=905.0, fwhm=22.0, model="pseudo_voigt", baseline_order=1)
fit.peaks()[0]
# {'amplitude': 1.6006, 'center': 905.005, 'sigma': 9.338, 'gamma': 12.05,
#  'eta': 0.003, 'fwhm': 21.996, 'area': 37.5,
#  'amplitude_err': 0.004, 'center_err': 0.012, ...}
# （数据为峰高 1.6、FWHM 22 的高斯峰，噪声水平 0.005）
fit.reduced_chi2, fit.converged     # (2.87e-05, True)

# 两个重叠峰“同时”拟合 —— 即重叠峰分解
fit = fit_region(x, y, centers=[100.0, 112.0], model="gaussian", baseline_order=1)
fit.curve()          # 峰形 + 背景
fit.background()     # 只取多项式背景

# 就地精修一整批检测结果
fit_peaks(result, model="gaussian")
```

拟合会给出参数标准误差、`reduced_chi2` 与 `converged` 标志；不需要手写雅可比矩阵（用中心
差分近似），并施加箱式约束（振幅与峰宽为正、峰位限制在拟合区间内、`eta ∈ [0, 1]`），因此
洛伦兹/Voigt 的长尾不会发散成退化解。

## 算法说明

**Savitzky–Golay** 系数通过把 `y` 投影到多项式基上求得（Vandermonde 矩阵的伪逆），因此平滑与
任意阶导数都支持。系数与滤波结果在内部区域与 SciPy 一致到 `1e-10`；边缘则取决于所选的填充
方式（默认 `reflect`），这是唯一可能与其他实现有差异的地方。

**基线**：airPLS 与 ALS 都是迭代重加权的 Whittaker 平滑，每次迭代求解

    (diag(w) + lambda * DᵀD) z = w ⊙ y

其中 `D` 是所选阶数的差分算子。该系统对称、正定且带状，因此这里用**带状 LDLᵀ 分解**求解，每次
迭代只需 `O(n)`，而不是稠密求解的 `O(n³)` —— 这正是 airPLS 能用在长光谱上的原因。SNIP 在对数-
对数-平方根域中迭代裁剪峰；形态学与滚球法属于非平坦结构元的开运算；多项式法则不断剔除高于当前
拟合值的点并重新拟合。

**prominence 与峰宽**完全遵循 SciPy 的定义：prominence 等于峰高减去两侧基准点中较高者，基准点由
「向最近的更高样点行走」确定（并列最小时与 SciPy 一样取**更靠后**的那个）；峰宽是
`y_peak - rel_height * prominence` 等高线与曲线的插值交点之间的距离。在随机信号与合成信号上，
峰索引与所有属性数组都与 `scipy.signal` 逐元素一致（见测试套件）。

**导数法**对 Savitzky–Golay 拟合结果求导（数值上稳定，避免直接对原始数据求导），寻找 `d1` 由正
转负且 `d2 < 0` 的过零点，或寻找 `d2` 的极小值（用于分解没有独立一阶导数过零点的肩峰）。峰位由
过零点线性插值给出，具备亚采样精度（测试中样点 500.37 处的峰定位到 ±0.001 样点），返回的索引则
吸附到真正的峰顶样点。

**小波法**计算 Ricker（墨西哥帽）小波变换并跨尺度跟踪**脊线**。脊点需超过 `snr` 倍该尺度的噪声
水平，噪声用稳健的 `1.4826 · MAD(|CWT|)` 估计；只有当脊线跨越至少 `min_length` 个尺度、并且位置
随尺度变化保持稳定时才被接受 —— 宽峰的旁瓣会随尺度向外漂移，因此被这一判据剔除。对高斯峰而言，
响应最强的尺度约等于其半高全宽（样点单位），据此选择 `widths` 范围（大致覆盖关心峰宽的 0.5～2 倍）。

**拟合**是朴素的 Levenberg–Marquardt（阻尼高斯-牛顿）迭代，使用中心差分雅可比与可选的箱式投影。
模型为多个峰形之和加多项式背景，背景在归一化坐标 `(x - x̄) / std(x)` 下求值，使跨数量级的 x 轴
也有良好的条件数。协方差取 `(JᵀJ)⁻¹ s²`，因此报告的误差就是通常意义上的 1σ 不确定度。

## 精度实测（由测试套件给出）

| 检验项 | 结果 |
| --- | --- |
| `find_peaks`、`peak_prominences`、`peak_widths` 对比 `scipy.signal` | 峰索引与全部属性数组完全一致（9 组滤波组合 × 25 条信号，另含 `rel_height` 0.25/0.5/0.9 与两组 `wlen`，并覆盖高度并列的情形） |
| Savitzky–Golay 系数/滤波 对比 `scipy.signal` | 内部区域一致到 `1e-10` |
| 带状求解 对比 稠密 `numpy.linalg.solve` | 相对误差 `1e-13`；2 万点 airPLS 单步约 0.1 s |
| 6 峰合成谱的峰位（噪声 σ=0.002，SNR 180–580，指数背景 + airPLS） | 三种探测器误差均 ≤ 0.3 样点，且无误检 |
| 同一光谱的峰宽（半高全宽） | 误差小于 1.7 % |
| 峰面积（x 轴单位） | 误差小于 6 % |
| 样点 500.37 处的峰定位 | 插值位置 500.370 ± 0.001 |
| 相隔 12 样点的两个高斯峰（σ≈6）的分离 | 峰位误差 0.12 样点，峰高误差 2 % |
| 用高斯模型拟合高斯数据 | 全部参数落在 1 个标准误差内 |

运行测试：

```bash
pytest              # 173 个用例；依赖 scipy 的用例在缺少 scipy 时自动跳过
pytest -q -k scipy  # 只跑交叉验证用例
```

## 约定与注意事项

* **x 轴单位 vs 样点**：`Peak.x`、`Peak.width`、`Peak.area` 使用 x 轴单位；`Peak.index`、
  `left_ip`、`right_ip`、`left_base`、`right_base` 使用样点索引。未提供 x 轴时两者相同，
  `x` 必须严格递增。
* **`rel_height`** 从峰顶向下、按 prominence 的比例计：0.5 给出半高全宽；0.25 是 75 % 高度处的宽度
  （更窄）；0.95 接近基线（更宽）。
* **`Peak.area`** 在 `±area_span/2 × width`（默认 ±1.5 倍半高全宽）的窗口内对处理后的信号积分，
  并减去窗口两端点连线。分辨良好的高斯型峰误差仅几个百分点；对洛伦兹/Voigt 这类长尾峰形会被截断；
  重叠峰请改用 `fit_peaks()` 给出的模型解析面积。
* **高度并列**：`distance` 滤波按 `numpy.argsort` 的顺序访问等高峰（与 SciPy 相同的并列处理），
  需要别的规则请自行先排序。
* **默认门限**：未指定滤波条件时，`detect_peaks` 只保留 prominence 大于
  `5 × 1.4826 · MAD(处理后信号)` 的峰。基线残留多时调高，需要弱峰时调低。基线校正不完美时还会留下
  **又低又宽**的残留隆起（它们确实很宽），任何 prominence 门限都挡不住，请用 `max_width` 去掉，如
  上面的示例所示。
* **滚球法**：球在数据坐标下各向同性，基线在弯曲处的偏高会随 `radius` 近似二次增长；半径取最宽峰
  半宽的几倍，并在无峰区抽查结果。
* **形态学开运算**（平窗）在倾斜背景上的偏高约为 `half_window × |背景斜率|`，这是方法本身的特性，
  测试套件已把它固定下来。
* **Voigt** 峰形要求均匀网格且样点数为奇数（该峰形用 FFT 卷积求值）。
* **拟合需要合理初值**：初值默认取自检测结果（通常已足够准）；若不理想，`fit_region` 允许直接指定
  峰位、峰宽与峰高。

## 目录结构

```
src/spectral_peaks/
    peaks.py        探测器、prominence/峰宽、detect_peaks 流程
    baseline.py     SNIP / airPLS / ALS / 形态学 / 滚球法 / 多项式
    smoothing.py    Savitzky-Golay、滑动平均、高斯平滑
    metrics.py      噪声、SNR、面积、FWHM 换算
    fitting.py      峰形模型 + Levenberg-Marquardt + 多峰拟合
    types.py        Peak / PeakResult 容器
    _banded.py      Whittaker 平滑所需的带状 LDL^T 线性代数
tests/              173 个用例，含与 SciPy 的交叉验证
```

尚未包含：命令行工具（CLI）、仪器格式文件的读写、合成光谱生成器、绘图辅助。目前可用
`PeakResult.to_csv()` 完成简单导出。

## 参考文献

（文献标题保留原文。）

* A. Savitzky, M.J.E. Golay, *Smoothing and Differentiation of Data by Simplified Least
  Squares Procedures*, Anal. Chem. 36 (1964) 1627.
* C.G. Ryan et al., *SNIP, a statistics-sensitive background treatment for the
  quantitative analysis of PIXE spectra*, Nucl. Instrum. Methods B 34 (1988) 396.
* P.H.C. Eilers, H.F.M. Boelens, *Baseline correction with asymmetric least squares
  smoothing*, Leiden University (2005).
* Z.-M. Zhang, S. Chen, Y.-Z. Liang, *Baseline correction using adaptive iteratively
  reweighted penalized least squares*, Analyst 135 (2010) 1138.
* S. Sternberg, *Biomedical image processing*, IEEE Computer 16 (1983) 22（形态学开运算；
  滚球法背景估计）.
* M.A. Kneen, H.J. Annegarn, *Algorithm for fitting XRF, SEM and PIXE X-ray spectra
  backgrounds*, Nucl. Instrum. Methods B 109 (1996) 209.
* P. Du, W.A. Kibbe, S.M. Lin, *Improved peak detection in mass spectrum by incorporating
  continuous wavelet transform-based pattern matching*, Bioinformatics 22 (2006) 2059.
* SciPy 的 `find_peaks`/`peak_prominences`/`peak_widths`（BSD-3）作为局部极值与 prominence 约定
  的参考实现；本项目的代码是独立的 NumPy 实现。

## 许可

MIT。

