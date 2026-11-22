"""Bảng phụ lục: bằng chứng giải thích theo từng truy vấn trên ViHeritage-MMKG (Cấp độ 3, đề mục 5: fidelity, validity,
sparsity cho tối thiểu 10 truy vấn). Mười truy vấn của mỗi mô hình (AdaMF, GAT), tách thành công và thất bại.

Nguồn: case_inputs/appendix_vh/D4_cases.md (chép nguyên results/MMKG/XAI_L3/D4_cases.md; K* = 1 với AdaMF, 8 với GAT).
Chạy: python report/figures/scripts/make_vh_cases_table.py   ->   report/Appendix/appendix_vh_cases_table.tex
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "case_inputs" / "appendix_vh" / "D4_cases.md"
OUT = HERE.parents[1] / "Appendix" / "appendix_vh_cases_table.tex"


def tex(s):
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ('"', "''")]:
        s = s.replace(a, b)
    return s


def num(x):
    return f"{x:.3f}".replace("-", "$-$")


def parse_triple(s):
    h, r, t = re.match(r"(.+?) –\[(.+?)\]→ (.+)", s.strip()).groups()
    return h.strip(), r.strip(), t.strip()


def read_vh(md):
    t = Path(md).read_text(encoding="utf-8")
    cases = []
    for blk in t.split("\n## ")[1:]:
        cid, arch, loai, seed = re.match(r"(\w+) — (\w+), (thành công|thất bại) \(seed (\d+)\)", blk).groups()
        q = re.search(r"Câu: “(.+?)”", blk).group(1)
        ans, top, rank = re.search(r"Đáp án: (.+?); ảnh hạng 1: \*\*(.+?)\*\*; ảnh đúng tốt nhất ở hạng (\d+)", blk).groups()
        S = []
        for line in re.findall(r"^\s+- (.+–\[.+\]→.+)$", blk, re.M):
            tr, d, rest = re.match(r"(.+?)\s+\(δ = ([-+.\d]+); (.*)\)\s*$", line).groups()
            S.append((parse_triple(tr), float(d), rest.startswith("đường đi")))
        f = re.search(r"Fid\+ ([-.\d]+) \(ngẫu nhiên ([-.\d]+)\), Fid− ([-.\d]+), PN (\d), PS (\d), làm nhiễu ([-.\d]+), "
                      r"sparsity ([.\d]+)", blk).groups()
        cases.append(dict(id=cid, arch=arch.lower(), loai=loai, second_run=seed != "2027", query=q, answer=ans, top=top,
                          rank=int(rank), S=S, fid=float(f[0]), fid_rand=float(f[1]), fid_minus=float(f[2]),
                          PN=int(f[3]), PS=int(f[4]), sparsity=float(f[6])))
    return cases


def table(cases):
    rows, prev = [], None
    order = sorted(cases, key=lambda c: (c["arch"] != "adamf", c["loai"] != "thành công", c["id"]))
    k = 0
    for c in order:
        grp = (c["arch"], c["loai"])
        if grp != prev:
            if prev is not None:
                rows.append(r"\midrule")
            model = "AdaMF" if c["arch"] == "adamf" else "GAT"
            rows.append(rf"\multicolumn{{11}}{{@{{}}l}}{{\textit{{CLIP + {model}, {c['loai']}}}}}\\")
            prev, k = grp, 0
        k += 1
        (h, r, t), d, _ = c["S"][0]
        on_path = sum(1 for _, _, p in c["S"] if p)
        mark = r"$^{\dagger}$" if c["second_run"] else ""
        rows.append(f"{k}{mark} & {tex(c['query'])} & {tex(c['answer'])} ({c['rank']}) & {tex(c['top'])} & "
                    f"{tex(h)} \\textit{{{tex(r)}}} {tex(t)} ({d:.3f}) & {len(c['S'])} & {on_path}/{len(c['S'])} & "
                    f"{num(c['fid'])} ({num(c['fid_rand'])}) & {num(c['fid_minus'])} & {c['PN']}/{c['PS']} & "
                    f"{c['sparsity']:.2f}\\\\")
    body = "\n".join(rows)
    return rf"""% Sinh bằng report/figures/scripts/make_vh_cases_table.py từ D4_cases.md.
\begin{{table*}}[!t]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{2.2pt}}
\caption{{Bằng chứng giải thích theo từng truy vấn trên ViHeritage-MMKG: mười truy vấn của mỗi mô hình}}
\label{{tab:app-vh-cases}}
\begin{{tabularx}}{{\linewidth}}{{@{{}}rL{{3.6cm}}L{{2.2cm}}L{{2.0cm}}Yrrrrrr@{{}}}}
\toprule
\# & Câu truy vấn & Đáp án (hạng) & Ảnh hạng 1 & Bộ ba giải thích mạnh nhất ($\delta$) & $|\mathcal S|$ & \twoline{{Đường}}{{đi}} & Fid+ & Fid$-$ & \twoline{{PN/}}{{PS}} & \twoline{{Spar-}}{{sity}}\\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\par\smallskip{{\noindent\raggedright\scriptsize Đáp án (hạng): hạng của ảnh đúng tốt nhất. $K^\star=1$ với AdaMF và $8$ với GAT. Đường đi: số bộ ba của $\mathcal S$ nằm trên đường đi từ thực thể neo tới thực thể của ảnh. Số trong ngoặc ở cột Fid+ là mức của bỏ cùng số bộ ba ngẫu nhiên. Mọi bộ ba giải thích là bộ ba train, nên validity cấu trúc đạt 100\%. $^{{\dagger}}$ Truy vấn lấy từ lần huấn luyện thứ hai; các truy vấn còn lại từ checkpoint chính.\par}}
\end{{table*}}
"""


if __name__ == "__main__":
    cases = read_vh(SRC)
    OUT.write_text(table(cases), encoding="utf-8")
    for a in ("adamf", "gat"):
        sub = [c for c in cases if c["arch"] == a]
        print(a, "Fid+ > ngẫu nhiên:", sum(c["fid"] > c["fid_rand"] for c in sub), "/", len(sub),
              "| PN=1:", sum(c["PN"] for c in sub), "| PS=1:", sum(c["PS"] for c in sub),
              "| sparsity", min(c["sparsity"] for c in sub), "đến", max(c["sparsity"] for c in sub),
              "| Fid+", min(c["fid"] for c in sub), "đến", max(c["fid"] for c in sub))
    print("đã ghi", OUT)
