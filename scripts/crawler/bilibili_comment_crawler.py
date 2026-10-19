# -*- coding: utf-8 -*-
"""
bilibili_comment_crawler.py
===========================
B 站视频评论爬虫（含 wbi 签名与二级回复爬取）。

功能说明
--------
1. 根据 BV 号自动解析视频 oid（aid）与标题；
2. 调用 B 站评论 wbi 主接口（/x/v2/reply/wbi/main）逐页爬取一级评论；
3. 支持可选爬取二级回复（/x/v2/reply/reply）；
4. 内置 w_rid 签名算法（wts + 固定 salt 的 MD5）、分页游标、限速休眠；
5. 结果以 CSV 落盘，字段含评论层级、用户等级、性别、点赞、回复数、IP 属地等。

使用前准备
----------
1. 在浏览器登录 B 站后，复制 Cookie 字符串保存到脚本同目录 `bili_cookie.txt`；
2. 修改下方 `VIDEO_BV` 为目标视频 BV 号；
3. 运行：python bilibili_comment_crawler.py

注意
----
仅用于学术研究，请控制爬取频率，遵守 B 站用户协议与 robots 规则；
本项目分析前已对结果数据做脱敏处理（删除用户 ID / 用户名 / 个性签名 / 头像）。
"""

import csv
import hashlib
import json
import re
import time
import urllib.parse

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# 配置区
# ---------------------------------------------------------------------------
VIDEO_BV = "BV1LJCaYaEET"   # 目标视频 BV 号
CRAWL_SECOND_LEVEL = True   # 是否爬取二级回复
COOKIE_FILE = "bili_cookie.txt"
OUTPUT_PREFIX_LEN = 12      # 输出文件名截取标题的最大长度

# wbi 签名固定 salt（B 站前端公开常量）
WBI_SALT = "ea1db124af3c7062474693fa704f4ff8"


def get_header() -> dict:
    """读取本地 Cookie 文件，构造带 UA 的请求头。"""
    with open(COOKIE_FILE, "r", encoding="utf-8") as f:
        cookie = f.read().strip()
    return {
        "Cookie": cookie,
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) "
            "Gecko/20100101 Firefox/127.0"
        ),
    }


def get_video_info(bv: str) -> tuple:
    """通过 BV 号请求视频页，提取评论接口所需的 oid（aid）与视频标题。"""
    resp = requests.get(
        f"https://www.bilibili.com/video/{bv}/", headers=get_header(), timeout=30
    )
    # oid 藏在页面 JSON 里："aid":xxxx,"bvid":"BVxxxx"
    obj = re.compile(rf'"aid":(?P<id>.*?),"bvid":"{bv}"')
    oid = obj.search(resp.text).group("id")

    obj_title = re.compile(r'<title data-vue-meta="true">(?P<title>.*?)</title>')
    match = obj_title.search(resp.text)
    title = match.group("title") if match else "未识别"
    return oid, title


def md5_sign(code: str) -> str:
    """对查询参数串做 MD5，生成 wbi 签名 w_rid。"""
    return hashlib.md5(code.encode("utf-8")).hexdigest()


def build_main_url(oid, page_id: str, wts: int) -> str:
    """构造评论主接口 URL（含 wbi 签名），page_id 为空表示第一页。"""
    plat, rtype, web_location = 1, 1, 1315875
    mode = 2  # 2=最新评论优先，3=热门评论优先
    if page_id:
        pagination_str = '{"offset":"%s"}' % page_id
        raw = (
            f"mode={mode}&oid={oid}&pagination_str={urllib.parse.quote(pagination_str)}"
            f"&plat={plat}&type={rtype}&web_location={web_location}&wts={wts}" + WBI_SALT
        )
    else:
        pagination_str = '{"offset":""}'
        raw = (
            f"mode={mode}&oid={oid}&pagination_str={urllib.parse.quote(pagination_str)}"
            f"&plat={plat}&seek_rpid=&type={rtype}&web_location={web_location}"
            f"&wts={wts}" + WBI_SALT
        )
    w_rid = md5_sign(raw)
    return (
        f"https://api.bilibili.com/x/v2/reply/wbi/main?oid={oid}&type={rtype}"
        f"&mode={mode}&pagination_str={urllib.parse.quote(pagination_str, safe=':')}"
        f"&plat={plat}&web_location={web_location}&w_rid={w_rid}&wts={wts}"
    )


def parse_reply(reply: dict, count: int, csv_writer) -> int:
    """解析单条评论（一级或二级），写入 CSV；返回更新后的计数。"""
    count += 1
    member = reply["member"]
    try:
        ip_location = reply["reply_control"]["location"][5:]
    except (KeyError, TypeError):
        ip_location = "未知"
    try:
        sub_text = reply["reply_control"]["sub_reply_entry_text"]
        reply_count = int(re.findall(r"\d+", sub_text)[0])
    except (KeyError, TypeError, IndexError):
        reply_count = 0

    csv_writer.writerow([
        count,
        reply["parent"],                                   # 上级评论ID
        reply["rpid"],                                     # 评论ID
        member["mid"],                                     # 用户ID（发布前需脱敏）
        member["uname"],                                   # 用户名（发布前需脱敏）
        member["level_info"]["current_level"],             # 用户等级
        member["sex"],                                     # 性别
        reply["content"]["message"],                       # 评论内容
        pd.to_datetime(reply["ctime"], unit="s"),          # 评论时间
        reply_count,                                       # 回复数
        reply["like"],                                     # 点赞数
        member.get("sign", ""),                            # 个性签名（发布前需脱敏）
        ip_location,                                       # IP属地
        "是" if member["vip"]["vipStatus"] else "否",       # 是否是大会员
        member["avatar"],                                  # 头像（发布前需脱敏）
    ])
    return count


def crawl_sub_replies(oid, root_rpid, pages: int, count: int, csv_writer) -> int:
    """爬取某条一级评论下的全部二级回复。"""
    for page in range(1, pages + 1):
        url = (
            f"https://api.bilibili.com/x/v2/reply/reply?oid={oid}&type=1"
            f"&root={root_rpid}&ps=10&pn={page}&web_location=333.788"
        )
        resp = json.loads(
            requests.get(url=url, headers=get_header(), timeout=30)
            .content.decode("utf-8")
        )
        if "data" not in resp:
            continue
        sub_replies = resp["data"].get("replies")
        if sub_replies:
            for sub in sub_replies:
                count = parse_reply(sub, count, csv_writer)
    return count


def crawl_page(bv, oid, page_id, count, csv_writer, crawl_second: bool):
    """爬取一页一级评论；返回 (bv, oid, 下一页游标, 计数, writer, 开关)。"""
    url = build_main_url(oid, page_id, wts=int(time.time()))
    data = json.loads(
        requests.get(url=url, headers=get_header(), timeout=30).content.decode("utf-8")
    )

    # 接口异常时打印原因并终止翻页，避免死循环
    if "data" not in data:
        print(f"一级评论接口无 data！返回内容：{data}")
        return bv, oid, 0, count, csv_writer, crawl_second
    if not data["data"].get("replies"):
        print("当前页无一级评论，爬取结束")
        return bv, oid, 0, count, csv_writer, crawl_second

    for reply in data["data"]["replies"]:
        count = parse_reply(reply, count, csv_writer)
        if count % 1000 == 0:
            time.sleep(20)          # 每 1000 条休眠 20s，规避反爬
        # 需要时爬取该评论的二级回复
        if crawl_second and reply.get("rcount", 0) != 0:
            pages = reply["rcount"] // 10 + 2
            count = crawl_sub_replies(oid, reply["rpid"], pages, count, csv_writer)

    # 解析下一页游标
    try:
        next_page_id = data["data"]["cursor"]["pagination_reply"]["next_offset"]
    except (KeyError, TypeError):
        next_page_id = 0

    if next_page_id == 0:
        print(f"评论爬取完成！总共爬取 {count} 条。")
    else:
        time.sleep(0.5)
        print(f"当前已爬取 {count} 条。")
    return bv, oid, next_page_id, count, csv_writer, crawl_second


def main() -> None:
    oid, title = get_video_info(VIDEO_BV)
    next_page_id = ""      # 空字符串表示从第一页开始
    count = 0

    out_file = f"{title[:OUTPUT_PREFIX_LEN]}_评论.csv"
    with open(out_file, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "序号", "上级评论ID", "评论ID", "用户ID", "用户名", "用户等级", "性别",
            "评论内容", "评论时间", "回复数", "点赞数", "个性签名", "IP属地",
            "是否是大会员", "头像",
        ])
        while next_page_id != 0:
            bv, oid, next_page_id, count, writer, CRAWL_SECOND_LEVEL = crawl_page(
                VIDEO_BV, oid, next_page_id, count, writer, CRAWL_SECOND_LEVEL
            )


if __name__ == "__main__":
    main()
