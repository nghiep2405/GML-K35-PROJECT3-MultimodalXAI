"""Generate the three level-specific XAI charts used by main_slide_vi.tex.

Only Python's standard library is required. Values and standard deviations are
the five rows already plotted by make_slide_charts.py, from report Table XIV.
Run from the repo root: python slide/make_xai_slide_charts.py
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'assets' / 'charts'
ROWS = {
    1: [('Visual Genome / GAT', .118, .002, .003, .0004)],
    2: [('DB15K / AdaMF', .511, .065, .023, .011),
        ('FB15K / AdaMF', .335, .041, .011, .044)],
    3: [('ViHeritage / AdaMF', .535, .059, .145, .032),
        ('ViHeritage / GAT', .103, .080, .002, .014)],
}


def make_chart(level, rows):
    lines = [r'\begin{tikzpicture}[x=1cm,y=4.7cm,font=\scriptsize]',
             r'\draw[Slate] (0,0) -- (0,0.68);',
             r'\draw[Slate] (0,0) -- (6.0,0);']
    for tick in [0, .2, .4, .6]:
        lines += [rf'\draw[LineGray] (0,{tick}) -- (6.0,{tick});',
                  rf'\node[left,text=Slate] at (0,{tick}) {{{tick:.1f}}};']
    lines.append(r'\node[rotate=90,text=Slate] at (-0.75,0.34) {Fid+};')
    centers = [3.0] if len(rows) == 1 else [1.65, 4.55]
    for center, (name, mean, sd, rand, rand_sd) in zip(centers, rows):
        for x, v, err, color in [(center-.40, mean, sd, 'HCMUSBlue'),
                                  (center+.40, rand, rand_sd, 'L2Orange')]:
            low, high = max(0, v-err), v+err
            lines += [rf'\fill[{color}] ({x-.32},0) rectangle ({x+.32},{v});',
                      rf'\draw[Slate] ({x},{low}) -- ({x},{high});',
                      rf'\draw[Slate] ({x-.10},{low}) -- ({x+.10},{low});',
                      rf'\draw[Slate] ({x-.10},{high}) -- ({x+.10},{high});',
                      rf'\node[above,text=Ink] at ({x},{high+.013}) {{{v:.3f}}};']
        lines.append(rf'\node[below,align=center,text=Slate] at ({center},-0.02) {{{name}}};')
    lines += [r'\fill[HCMUSBlue] (0.2,-0.17) rectangle (0.45,-0.135);',
              r'\node[right] at (0.5,-0.1525) {Bỏ tập giải thích $S$};',
              r'\fill[L2Orange] (3.2,-0.17) rectangle (3.45,-0.135);',
              r'\node[right] at (3.5,-0.1525) {Bỏ ngẫu nhiên cùng số phần tử};',
              r'\end{tikzpicture}']
    (OUT / f'xai_level{level}.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    for level, rows in ROWS.items():
        make_chart(level, rows)
    print('Generated three XAI chart sources.')
