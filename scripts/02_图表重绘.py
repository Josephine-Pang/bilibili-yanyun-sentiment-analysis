# -*- coding: utf-8 -*-
"""
02_图表重绘.py
================
《燕云十六声》B 站评论文化传播效果分析 —— 论文图表批量重绘脚本。

设计规范
--------
* 配色：莫兰迪色系 × 科技蓝调，全局统一色板，情感语义色（正面=莫兰迪绿、
  中性=雾蓝灰、负面=莫兰迪红）；
* 图表类型：渐变面积图、棒棒糖图、百分比堆叠条、小提琴图、雷达图、
  相关系数热力图、气泡散点图、哑铃图、渐变条形图、环形图、双向条形图；
* 输出：figures/ 目录下 fig01~fig11，统一 300 dpi 高清 PNG。

数据来源
--------
* data/processed/评论主数据集.csv      —— 评论级主数据（已脱敏，含情感与主题标注）
* data/processed/各视频传播效果排名.csv —— 视频粒度聚合指标
* data/processed/文化词汇词频表.csv     —— 文化词汇词频（词云素材）

使用方法
--------
    python scripts/02_图表重绘.py
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

# ---------------------------------------------------------------------------
# 路径与全局配置
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "figures"

# 全局配色：莫兰迪色系 × 科技蓝调
PALETTE = {
    "blue":   "#4A7BA6",   # 科技蓝（主色）
    "blue2":  "#7EA8C9",   # 浅科技蓝
    "green":  "#8FAE8B",   # 莫兰迪绿
    "red":    "#C98B8B",   # 莫兰迪红
    "grey":   "#A9B4BE",   # 雾蓝灰
    "sand":   "#B8A289",   # 暖沙棕
    "purple": "#9B8AA6",   # 雾紫
    "teal":   "#6FA3A0",   # 青灰绿
    "gold":   "#C9A96A",   # 哑金
    "dark":   "#2F3B47",   # 墨蓝（文字）
}
# 四类传播主题的固定配色
THEME_COLORS = {
    "文脉传承共振": PALETTE["blue"],
    "社会责任共振": PALETTE["green"],
    "科技赋能共振": PALETTE["teal"],
    "地域活化共振": PALETTE["sand"],
}
# 情感三分类语义色
SENTIMENT_COLORS = {"正面": PALETTE["green"], "中性": PALETTE["grey"], "负面": PALETTE["red"]}

THEME_ORDER = ["文脉传承共振", "社会责任共振", "科技赋能共振", "地域活化共振"]


def setup_style() -> None:
    """统一 matplotlib 全局风格：中文黑体、去冗余边框、浅网格、现代字体层级。"""
    mpl.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#D5DBE0",
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.color": "#E8ECEF",
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        "xtick.color": "#5A6570",
        "ytick.color": "#5A6570",
        "text.color": PALETTE["dark"],
        "axes.labelcolor": PALETTE["dark"],
        "axes.titlesize": 15,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def beautify(ax: plt.Axes, xgrid: bool = False, ygrid: bool = True) -> None:
    """去顶/右边框，按需保留网格线。"""
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x" if xgrid and not ygrid else ("y" if ygrid and not xgrid else "both"),
            alpha=0.8)
    if xgrid and ygrid:
        ax.grid(axis="both", alpha=0.6)


def save_fig(fig: plt.Figure, name: str) -> None:
    """统一导出 300 dpi PNG。"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"{name}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  已生成 {out.name}")


def gradient_fill(ax: plt.Axes, x, y, color: str, alpha_top: float = 0.40) -> None:
    """在折线下方绘制垂直渐变填充（顶部最深、底部透明）。"""
    # 日期轴需先转换为 matplotlib 数值坐标（天），否则 extent 与刻度尺度不一致
    if hasattr(x, "dt") or isinstance(x, (pd.DatetimeIndex,)):
        x = mdates.date2num(pd.DatetimeIndex(x))
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    line, = ax.plot(x, y, color=color, lw=1.8, zorder=3, solid_capstyle="round")

    z = np.empty((256, 1, 4), dtype=float)
    z[:, :, :3] = mcolors.to_rgb(color)
    z[:, :, 3] = np.linspace(0.02, alpha_top, 256)[:, None]
    xmin, xmax, ymin = x.min(), x.max(), y.min()
    im = ax.imshow(z, aspect="auto", extent=[xmin, xmax, ymin, y.max()],
                   origin="upper", zorder=2)
    xy = np.column_stack([x, y])
    xy = np.vstack([[xmin, ymin], xy, [xmax, ymin]])
    clip = Polygon(xy, closed=True, facecolor="none", edgecolor="none")
    ax.add_patch(clip)
    im.set_clip_path(clip)
    return line


# ---------------------------------------------------------------------------
# 图 1：评论声量时序趋势（渐变面积图 + 7 日移动平均）
# ---------------------------------------------------------------------------
def fig01_daily_trend(df: pd.DataFrame) -> None:
    daily = (
        df.assign(日期=pd.to_datetime(df["评论时间"]).dt.normalize())
          .groupby("日期").size()
    )
    # 补全无评论的日期并填 0，保证时序连续
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_idx, fill_value=0)
    ma7 = daily.rolling(7, center=True).mean()

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    gradient_fill(ax, daily.index, daily.values, PALETTE["blue"], alpha_top=0.38)
    ax.plot(ma7.index, ma7.values, color=PALETTE["red"], lw=2.0, ls="--",
            label="7 日移动平均", zorder=4)

    # 标注声量峰值
    peak_date, peak_val = daily.idxmax(), daily.max()
    ax.annotate(f"峰值 {peak_val} 条\n{peak_date:%Y-%m-%d}",
                xy=(peak_date, peak_val), xytext=(peak_date, peak_val * 1.13),
                ha="center", fontsize=10, color=PALETTE["dark"],
                arrowprops=dict(arrowstyle="-", color="#8A949E", lw=0.9))
    ax.scatter([peak_date], [peak_val], s=42, color=PALETTE["red"], zorder=5,
               edgecolor="white", lw=1.2)

    ax.set_title("图 1  《燕云十六声》B 站评论声量时序趋势", loc="left", pad=14, weight="bold")
    ax.set_xlabel("日期")
    ax.set_ylabel("日评论量（条）")
    ax.margins(x=0.015)
    ax.legend(loc="upper right")
    beautify(ax)
    save_fig(fig, "fig01_评论声量时序趋势")


# ---------------------------------------------------------------------------
# 图 2：各传播主题声量与互动（棒棒糖图，点大小 = 平均点赞）
# ---------------------------------------------------------------------------
def fig02_theme_lollipop(df: pd.DataFrame) -> None:
    stats = (
        df.groupby("传播主题")
          .agg(评论量=("评论内容", "size"), 平均点赞=("点赞数", "mean"))
          .reindex(THEME_ORDER)
          .sort_values("评论量")
    )
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    y = np.arange(len(stats))
    ax.hlines(y, 0, stats["评论量"], color="#C9D2D9", lw=2.6, zorder=2)
    sizes = 120 + (stats["平均点赞"] / stats["平均点赞"].max()) * 900
    scatter = ax.scatter(stats["评论量"], y, s=sizes,
                         c=[THEME_COLORS[t] for t in stats.index],
                         alpha=0.92, edgecolor="white", lw=1.6, zorder=3)
    for yi, (val, like) in enumerate(zip(stats["评论量"], stats["平均点赞"])):
        ax.text(val + stats["评论量"].max() * 0.015, yi,
                f"{val:,} 条 ｜ 均赞 {like:.1f}", va="center", fontsize=10,
                color=PALETTE["dark"])
    ax.set_yticks(y, stats.index, fontsize=11)
    ax.set_xlim(0, stats["评论量"].max() * 1.30)
    ax.set_title("图 2  各传播主题评论声量与互动水平", loc="left", pad=14, weight="bold")
    ax.set_xlabel("评论量（条，气泡大小 = 平均点赞数）")
    ax.set_ylabel("")
    beautify(ax, ygrid=False)
    save_fig(fig, "fig02_主题声量棒棒糖图")


# ---------------------------------------------------------------------------
# 图 3：情感结构构成（100% 堆叠横条）
# ---------------------------------------------------------------------------
def fig03_sentiment_structure(df: pd.DataFrame) -> None:
    ct = (
        pd.crosstab(df["传播主题"], df["情感分类"], normalize="index")
          .reindex(THEME_ORDER)[["正面", "中性", "负面"]] * 100
    )
    ct.loc["总体"] = (
        df["情感分类"].value_counts(normalize=True)[["正面", "中性", "负面"]] * 100
    )

    fig, ax = plt.subplots(figsize=(9.8, 4.9))
    left = np.zeros(len(ct))
    for col in ["正面", "中性", "负面"]:
        vals = ct[col].values
        ax.barh(ct.index, vals, left=left, height=0.62,
                color=SENTIMENT_COLORS[col], label=col, alpha=0.92,
                edgecolor="white", lw=1.0)
        for yi, (v, l) in enumerate(zip(vals, left)):
            if v >= 6:
                lum = "white" if col != "中性" else PALETTE["dark"]
                ax.text(l + v / 2, yi, f"{v:.1f}%", ha="center", va="center",
                        fontsize=9.5, color=lum)
        left += vals
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_title("图 3  各传播主题评论情感结构", loc="left", pad=14, weight="bold")
    ax.set_xlabel("占比")
    ax.set_ylabel("")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    beautify(ax, ygrid=False)
    save_fig(fig, "fig03_情感结构堆叠图")


# ---------------------------------------------------------------------------
# 图 4：情感深度分层（浅层正面 vs 深层情感 vs 负面，双向条形）
# ---------------------------------------------------------------------------
def fig04_emotion_depth(df: pd.DataFrame) -> None:
    # 深层情感 = 赞美认同 + 情感共鸣 + 文化自豪；浅层 = 一般正面
    deep = ["赞美认同", "情感共鸣", "文化自豪"]
    grp = df.copy()
    grp["层次"] = np.select(
        [grp["深度情感分类"].isin(deep), grp["深度情感分类"] == "一般正面",
         grp["深度情感分类"] == "负面"],
        ["深层情感", "浅层正面", "负面"], default="中性",
    )
    pct = (
        pd.crosstab(grp["传播主题"], grp["层次"], normalize="index")
          .reindex(THEME_ORDER)[["负面", "中性", "浅层正面", "深层情感"]] * 100
    )
    deep_pct = pct["深层情感"]

    fig, ax = plt.subplots(figsize=(9.8, 4.9))
    y = np.arange(len(pct))
    # 左侧堆叠：负面 + 中性（向左延伸）
    neg, neu = pct["负面"].values, pct["中性"].values
    ax.barh(y, -neg, height=0.60, color=SENTIMENT_COLORS["负面"], alpha=0.9,
            label="负面")
    ax.barh(y, -neu, left=-neg, height=0.60, color=SENTIMENT_COLORS["中性"],
            alpha=0.9, label="中性")
    # 右侧堆叠：浅层正面 + 深层情感
    ax.barh(y, pct["浅层正面"], height=0.60, color="#BCD3BC", alpha=0.95,
            label="浅层正面")
    ax.barh(y, deep_pct, left=pct["浅层正面"], height=0.60, color=PALETTE["blue"],
            alpha=0.95, label="深层情感（赞美/共鸣/自豪）")
    # 深层情感数值标注
    for yi, v in enumerate(deep_pct):
        ax.text(101.5, yi, f"深层 {v:.1f}%", va="center", fontsize=10,
                color=PALETTE["blue"], weight="bold")

    ax.axvline(0, color="#9AA5AF", lw=1.0)
    ax.set_yticks(y, pct.index, fontsize=11)
    ax.set_xlim(-100, 122)
    ax.set_xticks([-100, -50, 0, 50, 100],
                  ["100%", "50%", "0", "50%", "100%"])
    ax.set_title("图 4  各传播主题情感深度分层（左：负面与中性；右：正面）",
                 loc="left", pad=14, weight="bold")
    ax.set_ylabel("")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    beautify(ax, ygrid=False)
    save_fig(fig, "fig04_情感深度双向条形图")


# ---------------------------------------------------------------------------
# 图 5：情感得分分布（小提琴 + 内嵌箱线）
# ---------------------------------------------------------------------------
def fig05_sentiment_violin(df: pd.DataFrame) -> None:
    data = [df.loc[df["传播主题"] == t, "情感得分"].dropna() for t in THEME_ORDER]
    fig, ax = plt.subplots(figsize=(9.8, 5.0))
    vp = ax.violinplot(data, positions=range(len(data)), widths=0.78,
                       showmeans=False, showextrema=False)
    for body, t in zip(vp["bodies"], THEME_ORDER):
        body.set_facecolor(THEME_COLORS[t])
        body.set_alpha(0.55)
        body.set_edgecolor("white")
    bp = ax.boxplot(data, positions=range(len(data)), widths=0.16, patch_artist=True,
                    medianprops=dict(color="white", lw=1.6),
                    flierprops=dict(marker="", linestyle="none"),
                    boxprops=dict(edgecolor="none"), whiskerprops=dict(color="#8A949E"),
                    capprops=dict(color="#8A949E"), zorder=3)
    for box in bp["boxes"]:
        box.set_facecolor(PALETTE["dark"])
    ax.axhline(0, color="#B8C0C8", lw=0.9, ls=":", zorder=1)
    ax.set_xticks(range(len(THEME_ORDER)), THEME_ORDER, fontsize=11)
    ax.set_title("图 5  各传播主题情感得分分布", loc="left", pad=14, weight="bold")
    ax.set_xlabel("传播主题")
    ax.set_ylabel("情感得分（-1 ~ 1）")
    beautify(ax, ygrid=False)
    save_fig(fig, "fig05_情感得分小提琴图")


# ---------------------------------------------------------------------------
# 图 6：多维传播效果雷达图
# ---------------------------------------------------------------------------
def fig06_radar(df: pd.DataFrame) -> None:
    def safe(series: pd.Series) -> float:
        return series.mean() if len(series) else 0.0

    dims = ["正面率", "情感得分", "深度情感率", "行为意向率", "高赞互动率"]
    rows = {}
    for t in THEME_ORDER:
        sub = df[df["传播主题"] == t]
        rows[t] = [
            (sub["情感分类"] == "正面").mean() * 100,
            sub["情感得分"].mean(),
            sub["深度情感分类"].isin(["赞美认同", "情感共鸣", "文化自豪"]).mean() * 100,
            (sub["行为意向"] != "无明确意向").mean() * 100,
            (sub["点赞数"] >= 10).mean() * 100,
        ]
    mat = pd.DataFrame(rows, index=dims).T
    # 各维度 min-max 归一化（0.15~1.0，保留差异又不塌缩到圆心）
    norm = (mat - mat.min()) / (mat.max() - mat.min() + 1e-9) * 0.85 + 0.15

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.6, 7.2), subplot_kw=dict(polar=True))
    for t in THEME_ORDER:
        vals = norm.loc[t].tolist() + [norm.loc[t].iloc[0]]
        ax.plot(angles, vals, color=THEME_COLORS[t], lw=2.0, label=t)
        ax.fill(angles, vals, color=THEME_COLORS[t], alpha=0.10)
    ax.set_xticks(angles[:-1], dims, fontsize=11)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0], ["25", "50", "75", "100"], fontsize=8.5,
                  color="#8A949E")
    ax.set_ylim(0, 1.05)
    ax.grid(color="#E3E8EC")
    ax.spines["polar"].set_color("#D5DBE0")
    ax.set_title("图 6  各传播主题多维传播效果雷达图（归一化）",
                 pad=26, weight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2)
    save_fig(fig, "fig06_主题效果雷达图")


# ---------------------------------------------------------------------------
# 图 7：评论互动特征相关性热力图（Spearman）
# ---------------------------------------------------------------------------
def fig07_corr_heatmap(df: pd.DataFrame) -> None:
    d = pd.DataFrame({
        "点赞数": df["点赞数"],
        "回复数": df["回复数"],
        "用户等级": df["用户等级"],
        "大会员": (df["是否是大会员"] == "是").astype(int),
        "情感得分": df["情感得分"],
        "评论长度": df["评论内容"].astype(str).str.len(),
    })
    corr = d.corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "morandi_div", [PALETTE["blue"], "#F4F6F7", PALETTE["red"]]
    )
    im = ax.imshow(corr.where(~mask), cmap=cmap, vmin=-1, vmax=1)
    ax.grid(False)
    labels = list(corr.columns)
    ax.set_xticks(range(len(labels)), labels, rotation=32, ha="right", fontsize=10.5)
    ax.set_yticks(range(len(labels)), labels, fontsize=10.5)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if not mask[i, j]:
                v = corr.iloc[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9.5,
                        color="white" if abs(v) > 0.55 else PALETTE["dark"])
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_visible(False)
    ax.set_title("图 7  评论互动特征 Spearman 相关系数矩阵", loc="left", pad=14,
                 weight="bold")
    save_fig(fig, "fig07_互动特征相关性热力图")


# ---------------------------------------------------------------------------
# 图 8：视频级传播效果气泡散点图
# ---------------------------------------------------------------------------
def fig08_video_bubble(df: pd.DataFrame) -> None:
    vid = pd.read_csv(DATA_DIR / "各视频传播效果排名.csv")
    # 与主数据集的主题命名保持一致
    counts = df.groupby("视频名称").size().rename("实际评论量")
    vid["评论数量"] = vid["视频名称"].map(counts).fillna(vid["评论数量"])
    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    for t in THEME_ORDER:
        sub = vid[vid["传播主题"] == t]
        if sub.empty:
            continue
        ax.scatter(sub["评论数量"], sub["正面率"],
                   s=90 + sub["平均点赞数"].clip(upper=200) * 3.2,
                   c=THEME_COLORS[t], alpha=0.82, edgecolor="white", lw=1.5,
                   label=t, zorder=3)
    # 仅标注代表性视频，避免文字重叠
    labeled = vid.nlargest(6, "评论数量").index.tolist()
    labeled += vid.nlargest(2, "正面率").index.tolist()
    for i in dict.fromkeys(labeled):
        row = vid.loc[i]
        short = str(row["视频名称"])[:9]
        ax.annotate(short, (row["评论数量"], row["正面率"]),
                    textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=8.8, color="#5A6570")
    ax.set_title("图 8  视频级传播效果气泡图（气泡大小 = 平均点赞数）",
                 loc="left", pad=14, weight="bold")
    ax.set_xlabel("评论量（条）")
    ax.set_ylabel("正面率（%）")
    ax.legend(loc="upper right", title="传播主题", title_fontsize=10)
    beautify(ax, ygrid=True)
    save_fig(fig, "fig08_视频效果气泡图")


# ---------------------------------------------------------------------------
# 图 9：官方账号 vs 个人博主 多指标哑铃图
# ---------------------------------------------------------------------------
def fig09_official_vs_personal(df: pd.DataFrame) -> None:
    metrics = {
        "正面率(%)": lambda d: (d["情感分类"] == "正面").mean() * 100,
        "负面率(%)": lambda d: (d["情感分类"] == "负面").mean() * 100,
        "深度情感率(%)": lambda d: d["深度情感分类"]
            .isin(["赞美认同", "情感共鸣", "文化自豪"]).mean() * 100,
        "行为意向率(%)": lambda d: (d["行为意向"] != "无明确意向").mean() * 100,
        "高赞率(%)": lambda d: (d["点赞数"] >= 10).mean() * 100,
    }
    rows = []
    for name, fn in metrics.items():
        official = fn(df[df["账号类型"] == "官方账号"])
        personal = fn(df[df["账号类型"] == "个人博主"])
        rows.append([name, personal, official])
    res = pd.DataFrame(rows, columns=["指标", "个人博主", "官方账号"])

    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    y = np.arange(len(res))
    for yi, (_, r) in enumerate(res.iterrows()):
        lo, hi = min(r["个人博主"], r["官方账号"]), max(r["个人博主"], r["官方账号"])
        ax.hlines(yi, lo, hi, color="#C9D2D9", lw=3.2, zorder=2)
        ax.scatter(r["个人博主"], yi, s=150, color=PALETTE["gold"], zorder=3,
                   edgecolor="white", lw=1.4, label="个人博主" if yi == 0 else None)
        ax.scatter(r["官方账号"], yi, s=150, color=PALETTE["blue"], zorder=3,
                   edgecolor="white", lw=1.4, label="官方账号" if yi == 0 else None)
        gap = r["官方账号"] - r["个人博主"]
        ax.text(max(lo, hi) + res[["个人博主", "官方账号"]].max().max() * 0.03, yi,
                f"{gap:+.1f}pp", va="center", fontsize=9.5,
                color=PALETTE["blue"] if gap > 0 else PALETTE["gold"])
    ax.set_yticks(y, res["指标"], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, res[["个人博主", "官方账号"]].max().max() * 1.22)
    ax.set_title("图 9  官方账号与个人博主传播效果对比（标签为官方 - 个人差值）",
                 loc="left", pad=14, weight="bold")
    ax.set_xlabel("指标值")
    ax.legend(loc="lower right")
    beautify(ax, ygrid=False)
    save_fig(fig, "fig09_官方个人哑铃图")


# ---------------------------------------------------------------------------
# 图 10：文化词汇词频 TOP20（渐变条形图）
# ---------------------------------------------------------------------------
def fig10_cultural_words() -> None:
    words = pd.read_csv(DATA_DIR / "文化词汇词频表.csv").nlargest(20, "频次")
    words = words.iloc[::-1]  # 从下到上升序排列

    fig, ax = plt.subplots(figsize=(9.2, 7.6))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "tech_blue", ["#DCE7F1", PALETTE["blue"], PALETTE["dark"]]
    )
    norm = mcolors.Normalize(vmin=words["频次"].min(), vmax=words["频次"].max())
    colors = [cmap(norm(v)) for v in words["频次"]]
    ax.barh(words["词语"], words["频次"], color=colors, height=0.68,
            edgecolor="white", lw=0.6, zorder=3)
    for yi, v in enumerate(words["频次"]):
        ax.text(v + words["频次"].max() * 0.012, yi, str(int(v)),
                va="center", fontsize=9.5, color=PALETTE["dark"])
    ax.set_title("图 10  文化词汇词频 TOP 20", loc="left", pad=14, weight="bold")
    ax.set_xlabel("词频（次）")
    ax.set_ylabel("")
    ax.set_xlim(0, words["频次"].max() * 1.10)
    beautify(ax, ygrid=False)
    save_fig(fig, "fig10_文化词汇词频图")


# ---------------------------------------------------------------------------
# 图 11：行为意向类型分布（环形图）
# ---------------------------------------------------------------------------
def fig11_behavior_intention(df: pd.DataFrame) -> None:
    intents = (
        df.loc[df["行为意向"] != "无明确意向", "行为意向"]
          .value_counts()
    )
    intent_colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["gold"],
                     PALETTE["purple"], PALETTE["green"]][: len(intents)]

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    wedges, _ = ax.pie(
        intents.values, startangle=90, counterclock=False,
        colors=intent_colors, wedgeprops=dict(width=0.42, edgecolor="white", lw=2),
    )
    total = intents.sum()
    ax.text(0, 0.08, f"{total}", ha="center", va="center", fontsize=26,
            weight="bold", color=PALETTE["dark"])
    ax.text(0, -0.16, "条行为意向评论", ha="center", va="center",
            fontsize=10.5, color="#5A6570")
    legend_labels = [f"{name}  {v} 条（{v / total * 100:.1f}%）"
                     for name, v in intents.items()]
    ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=10.5)
    ax.set_title("图 11  用户行为意向类型分布", loc="left", pad=16, weight="bold")
    save_fig(fig, "fig11_行为意向环形图")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    setup_style()
    print("读取数据 ……")
    df = pd.read_csv(DATA_DIR / "评论主数据集.csv")
    df["评论时间"] = pd.to_datetime(df["评论时间"], utc=True).dt.tz_localize(None)
    print(f"  主数据集 {len(df)} 条评论，{df['传播主题'].nunique()} 类主题\n")

    print("开始绘制图表（输出目录：figures/）")
    fig01_daily_trend(df)
    fig02_theme_lollipop(df)
    fig03_sentiment_structure(df)
    fig04_emotion_depth(df)
    fig05_sentiment_violin(df)
    fig06_radar(df)
    fig07_corr_heatmap(df)
    fig08_video_bubble(df)
    fig09_official_vs_personal(df)
    fig10_cultural_words()
    fig11_behavior_intention(df)
    print("\n全部图表重绘完成，共 11 张，300 dpi 高清 PNG。")


if __name__ == "__main__":
    main()
