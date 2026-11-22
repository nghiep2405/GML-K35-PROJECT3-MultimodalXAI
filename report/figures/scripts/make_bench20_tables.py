"""Bảng phụ lục: bộ 20 truy vấn đánh giá của Cấp độ 2 (DB15K, FB15K-237-IMG), tách nhóm trực tiếp và gián tiếp.

Nguồn: our_run_result/results/{DB15K,FB15K}/XAI/C3_benchmark_queries.csv (hạng ảnh đúng của CLIP và của CLIP + AdaMF
ở ba lần huấn luyện). Thành công: ảnh đúng trong top 10 ở cả ba lần huấn luyện (như R@10). So với CLIP: hạng sau khi thêm
AdaMF so với hạng chỉ dùng CLIP, xét ở cả ba lần huấn luyện.
Chạy: python report/figures/scripts/make_bench20_tables.py   ->   report/Appendix/appendix_bench20_tables.tex
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RES = REPO / "our_run_result" / "results"
OUT = REPO / "report" / "Appendix" / "appendix_bench20_tables.tex"


def tex(s):
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ('"', "''")]:
        s = s.replace(a, b)
    return s


def compare(clip, ranks):
    sign = {("t" if r < clip else "k" if r > clip else "n") for r in ranks}
    if sign == {"t"}:
        return "tốt hơn"
    if sign == {"k"}:
        return "kém hơn"
    if sign == {"n"}:
        return r"như nhau$^{*}$" if clip > 1000 else "như nhau"
    return "lẫn lộn"


def table(ds, name, label):
    df = pd.read_csv(RES / ds / "XAI" / "C3_benchmark_queries.csv")
    rows = []
    for title, sub in (("Truy vấn trực tiếp", df[df.prediction_direction == "direct"]),
                       ("Truy vấn gián tiếp", df[df.prediction_direction != "direct"])):
        rows.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{title}}}}}\\")
        for i, r in enumerate(sub.itertuples(), 1):
            ranks = [r.hybrid_rank_seed2027, r.hybrid_rank_seed2028, r.hybrid_rank_seed2029]
            ada = str(ranks[0]) if len(set(ranks)) == 1 else "/".join(map(str, ranks))
            ok = "Thành công" if all(x <= 10 for x in ranks) else "Thất bại"
            chieu = "" if r.prediction_direction == "direct" else r.prediction_direction
            rows.append(f"{i} & {tex(r.query_text)} & {chieu} & {r.clip_rank} & {ada} & {compare(r.clip_rank, ranks)} & {ok}\\\\")
        rows.append(r"\midrule")
    body = "\n".join(rows[:-1])
    return rf"""\begin{{table*}}[!t]
\centering
\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\caption{{Bộ 20 truy vấn đánh giá trên {name}: hạng của ảnh đúng và kết quả từng truy vấn}}
\label{{{label}}}
\begin{{tabularx}}{{\linewidth}}{{@{{}}rYlrrll@{{}}}}
\toprule
\# & Truy vấn & Chiều & CLIP & CLIP + AdaMF & So với CLIP & Kết quả\\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\par\smallskip{{\noindent\raggedright\footnotesize Cột CLIP + AdaMF ghi một số khi ba lần huấn luyện cho cùng hạng, ba số khi khác nhau. Thành công: ảnh đúng nằm trong top 10 ở cả ba lần huấn luyện. Lẫn lộn: tốt hơn CLIP ở một số lần, kém hơn ở lần khác. $^{{*}}$ Ảnh đúng nằm ngoài top 1\,000 của CLIP nên không được xếp hạng lại.\par}}
\end{{table*}}
"""


if __name__ == "__main__":
    OUT.write_text("% Sinh bằng report/figures/scripts/make_bench20_tables.py từ C3_benchmark_queries.csv.\n"
                   + table("DB15K", "DB15K", "tab:app-bench20-db15k") + "\n"
                   + table("FB15K", "FB15K-237-IMG", "tab:app-bench20-fb15k"), encoding="utf-8")
    print("đã ghi", OUT)
