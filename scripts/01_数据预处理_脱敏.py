# -*- coding: utf-8 -*-
"""
01_数据预处理_脱敏.py
=====================
对 B 站评论原始爬取数据进行**脱敏清洗**，生成可直接用于分析与开源发布的结构化数据。

功能说明
--------
1. 逐视频读取 data/raw_source/ 下的原始爬取文件（.csv / .xlsx，14 个视频）；
2. 删除用户隐私字段：用户ID、用户名、个性签名、头像 —— 仅保留统计分析所需的
   匿名化互动字段（评论ID、等级、性别、内容、时间、点赞、回复、IP属地、大会员、BV号）；
3. 统一字段类型（评论时间 → datetime），统一输出为 UTF-8-SIG 编码的 CSV；
4. 处理汇总表 `数据汇总.xlsx`（个人账号 / 官方 两个 sheet），脱敏后另存；
5. 处理评论级主分析数据集（含情感得分、传播主题等衍生字段），脱敏后另存。

使用方法
--------
    python scripts/01_数据预处理_脱敏.py

输入：data/raw_source/   （整理前的原始爬取数据备份目录）
输出：data/raw/          （脱敏后的逐视频原始数据）
      data/processed/    （清洗后的结构化分析数据）
"""

from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent      # 项目根目录
RAW_SOURCE_DIR = PROJECT_ROOT / "data" / "raw_source"      # 原始数据（待脱敏）
RAW_OUT_DIR = PROJECT_ROOT / "data" / "raw"                # 脱敏后逐视频数据
PROCESSED_OUT_DIR = PROJECT_ROOT / "data" / "processed"    # 清洗后分析数据

# 需要从数据中删除的隐私字段（脱敏白名单之外的用户身份信息）
PRIVACY_COLUMNS = ["用户ID", "用户名", "昵称", "个性签名", "头像"]


def desensitize(df: pd.DataFrame) -> pd.DataFrame:
    """删除隐私字段，并返回仅保留剩余列的副本。"""
    drop_cols = [c for c in PRIVACY_COLUMNS if c in df.columns]
    return df.drop(columns=drop_cols)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """统一字段类型：评论时间转 datetime，数值列转数值型。"""
    df = df.copy()
    if "评论时间" in df.columns:
        df["评论时间"] = pd.to_datetime(df["评论时间"], errors="coerce")
    for col in ["点赞数", "回复数", "用户等级"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_any(path: Path) -> pd.DataFrame:
    """按扩展名自动选择 pandas 读取方式（csv/xlsx）。"""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    return pd.read_excel(path)


def main() -> None:
    RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"未找到原始数据目录：{RAW_SOURCE_DIR}\n"
            "请将整理前的原始爬取数据放入 data/raw_source/ 后重新运行。"
        )

    # -----------------------------------------------------------------------
    # 1. 逐视频脱敏 → data/raw/*.csv
    # -----------------------------------------------------------------------
    video_files = sorted(
        p for p in RAW_SOURCE_DIR.iterdir()
        if p.suffix.lower() in (".csv", ".xlsx") and p.stem[0].isdigit()
    )
    print(f"[1/3] 发现 {len(video_files)} 个逐视频原始文件，开始脱敏……")
    for path in video_files:
        df = normalize(desensitize(load_any(path)))
        out_path = RAW_OUT_DIR / (path.stem + ".csv")
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"      {path.name}  ->  {out_path.name}  （{len(df)} 条）")

    # -----------------------------------------------------------------------
    # 2. 汇总表（个人账号 / 官方）脱敏 → data/raw/数据汇总_脱敏.xlsx
    # -----------------------------------------------------------------------
    summary_path = RAW_SOURCE_DIR / "数据汇总.xlsx"
    if summary_path.exists():
        print("[2/3] 处理汇总表 数据汇总.xlsx（个人账号 / 官方）……")
        xls = pd.ExcelFile(summary_path)
        with pd.ExcelWriter(
            RAW_OUT_DIR / "数据汇总_脱敏.xlsx", engine="openpyxl"
        ) as writer:
            for sheet in xls.sheet_names:
                df = normalize(desensitize(pd.read_excel(xls, sheet_name=sheet)))
                df.to_excel(writer, sheet_name=sheet, index=False)
                print(f"      sheet[{sheet}]  {len(df)} 条，已脱敏")
    else:
        print("[2/3] 未找到 数据汇总.xlsx，跳过。")

    # -----------------------------------------------------------------------
    # 3. 评论级主分析数据集脱敏 → data/processed/评论主数据集.csv
    # -----------------------------------------------------------------------
    master_candidates = list(RAW_SOURCE_DIR.glob("*完整数据*.csv"))
    if master_candidates:
        master_path = master_candidates[0]
        print(f"[3/3] 处理评论级主分析数据集 {master_path.name} ……")
        df = normalize(desensitize(pd.read_csv(master_path, encoding="utf-8-sig")))
        df = df.dropna(subset=["评论内容"])          # 剔除空评论
        out = PROCESSED_OUT_DIR / "评论主数据集.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"      输出 {out.name}（{len(df)} 条，{len(df.columns)} 列）")
    else:
        print("[3/3] 未找到评论级主分析数据集，跳过。")

    print("\n脱敏与清洗完成：")
    print(f"  - 逐视频原始数据（脱敏）  -> {RAW_OUT_DIR}")
    print(f"  - 清洗后分析数据          -> {PROCESSED_OUT_DIR}")


if __name__ == "__main__":
    main()
