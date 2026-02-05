"""
PlotNeuralNet diagram for XEC Multi-Branch Architecture

Requirements:
    1. Clone PlotNeuralNet: git clone https://github.com/HarisIqbal88/PlotNeuralNet.git
    2. Install LaTeX with TikZ
    3. Run: python xec_architecture.py
    4. Compile: pdflatex xec_architecture.tex

Architecture Overview:
    Input (4760 sensors × 2 channels)
    → 6 Face Encoders (ConvNeXt for rect, HexNeXt for hex)
    → 6 × 1024-dim tokens
    → Transformer Fusion (2 layers, 8 heads)
    → Task Heads (Angle, Energy, Timing, Position)
"""

import sys
import os

# Add PlotNeuralNet to path - adjust this path as needed
# sys.path.append('/path/to/PlotNeuralNet')

# Since PlotNeuralNet might not be installed, we'll generate raw TikZ code directly


def generate_xec_tikz():
    """Generate TikZ code for XEC architecture diagram."""

    tikz_code = r"""
\documentclass[border=15pt, multi, tikz]{standalone}
\usepackage{import}
\usepackage{xcolor}

% Define custom colors
\definecolor{ConvColor}{RGB}{255,213,79}      % Yellow for ConvNeXt
\definecolor{HexColor}{RGB}{129,199,132}       % Green for HexNeXt
\definecolor{TransColor}{RGB}{100,181,246}     % Blue for Transformer
\definecolor{HeadColor}{RGB}{240,98,146}       % Pink for Task Heads
\definecolor{InputColor}{RGB}{189,189,189}     % Gray for Input
\definecolor{TokenColor}{RGB}{186,104,200}     % Purple for Tokens

\usetikzlibrary{positioning, shapes, arrows.meta, calc, fit, backgrounds}

\begin{document}
\begin{tikzpicture}[
    node distance=0.8cm,
    box/.style={rectangle, draw, rounded corners, minimum height=1cm, minimum width=2cm, align=center, font=\small},
    input/.style={box, fill=InputColor!30},
    conv/.style={box, fill=ConvColor!50},
    hex/.style={box, fill=HexColor!50},
    token/.style={circle, draw, fill=TokenColor!30, minimum size=0.8cm, font=\tiny},
    trans/.style={box, fill=TransColor!50, minimum width=4cm},
    head/.style={box, fill=HeadColor!50},
    arrow/.style={-{Stealth[length=2mm]}, thick},
    label/.style={font=\scriptsize, align=center},
]

% ============================================
% INPUT LAYER
% ============================================
\node[input, minimum width=3cm, minimum height=1.5cm] (input) {
    \textbf{Input}\\
    4760 sensors $\times$ 2 ch\\
    (Npho, Time)
};

% ============================================
% FACE ENCODERS - Rectangular (ConvNeXt)
% ============================================
\node[conv, right=2cm of input, yshift=2.5cm] (inner) {
    \textbf{Inner Face}\\
    93 $\times$ 44\\
    ConvNeXt V2
};

\node[conv, below=0.3cm of inner] (outer) {
    \textbf{Outer Face}\\
    9 $\times$ 24 (+ fine)\\
    ConvNeXt V2
};

\node[conv, below=0.3cm of outer] (us) {
    \textbf{Upstream}\\
    24 $\times$ 6\\
    ConvNeXt V2
};

\node[conv, below=0.3cm of us] (ds) {
    \textbf{Downstream}\\
    24 $\times$ 6\\
    ConvNeXt V2
};

% ============================================
% FACE ENCODERS - Hexagonal (HexNeXt)
% ============================================
\node[hex, below=0.3cm of ds] (top) {
    \textbf{Top PMT}\\
    334 hexagonal\\
    HexNeXt GAT
};

\node[hex, below=0.3cm of top] (bot) {
    \textbf{Bottom PMT}\\
    334 hexagonal\\
    HexNeXt GAT
};

% ============================================
% 1024-dim TOKENS
% ============================================
\node[token, right=1.5cm of inner] (t1) {1024};
\node[token, right=1.5cm of outer] (t2) {1024};
\node[token, right=1.5cm of us] (t3) {1024};
\node[token, right=1.5cm of ds] (t4) {1024};
\node[token, right=1.5cm of top] (t5) {1024};
\node[token, right=1.5cm of bot] (t6) {1024};

% ============================================
% TRANSFORMER FUSION
% ============================================
\node[trans, right=1.5cm of us, minimum height=4cm] (transformer) {
    \textbf{Transformer Encoder}\\[3pt]
    2 layers, 8 heads\\
    6 tokens $\rightarrow$ 6144-dim
};

% ============================================
% TASK HEADS
% ============================================
\node[head, right=2cm of transformer, yshift=1.5cm] (angle) {
    \textbf{Angle Head}\\
    $\theta$, $\phi$
};

\node[head, below=0.3cm of angle] (energy) {
    \textbf{Energy Head}\\
    E [MeV]
};

\node[head, below=0.3cm of energy] (timing) {
    \textbf{Timing Head}\\
    t [ns]
};

\node[head, below=0.3cm of timing] (position) {
    \textbf{Position Head}\\
    u, v, w, FI
};

% ============================================
% ARROWS - Input to Encoders
% ============================================
\draw[arrow] (input.east) -- ++(0.5,0) |- (inner.west);
\draw[arrow] (input.east) -- ++(0.5,0) |- (outer.west);
\draw[arrow] (input.east) -- ++(0.5,0) |- (us.west);
\draw[arrow] (input.east) -- ++(0.5,0) |- (ds.west);
\draw[arrow] (input.east) -- ++(0.5,0) |- (top.west);
\draw[arrow] (input.east) -- ++(0.5,0) |- (bot.west);

% ============================================
% ARROWS - Encoders to Tokens
% ============================================
\draw[arrow] (inner.east) -- (t1.west);
\draw[arrow] (outer.east) -- (t2.west);
\draw[arrow] (us.east) -- (t3.west);
\draw[arrow] (ds.east) -- (t4.west);
\draw[arrow] (top.east) -- (t5.west);
\draw[arrow] (bot.east) -- (t6.west);

% ============================================
% ARROWS - Tokens to Transformer
% ============================================
\draw[arrow] (t1.east) -- (t1.east -| transformer.west);
\draw[arrow] (t2.east) -- (t2.east -| transformer.west);
\draw[arrow] (t3.east) -- (t3.east -| transformer.west);
\draw[arrow] (t4.east) -- (t4.east -| transformer.west);
\draw[arrow] (t5.east) -- (t5.east -| transformer.west);
\draw[arrow] (t6.east) -- (t6.east -| transformer.west);

% ============================================
% ARROWS - Transformer to Heads
% ============================================
\draw[arrow] (transformer.east) -- ++(0.5,0) |- (angle.west);
\draw[arrow] (transformer.east) -- ++(0.5,0) |- (energy.west);
\draw[arrow] (transformer.east) -- ++(0.5,0) |- (timing.west);
\draw[arrow] (transformer.east) -- ++(0.5,0) |- (position.west);

% ============================================
% LABELS
% ============================================
\node[label, above=0.1cm of inner, xshift=1cm] {\textit{Rectangular faces}};
\node[label, above=0.1cm of top, xshift=1cm] {\textit{Hexagonal faces}};

% Architecture label
\node[font=\large\bfseries, above=1cm of transformer] {XEC Multi-Branch Architecture};

% Encoder details (optional)
\node[label, below=0.5cm of bot, xshift=-1cm, text width=4cm] {
    \textbf{ConvNeXt V2:}\\
    Stem $\rightarrow$ 2$\times$Block(32) $\rightarrow$\\
    Downsample $\rightarrow$ 3$\times$Block(64)
};

\node[label, right=0.3cm of t6, text width=3cm] {
    \textbf{HexNeXt:}\\
    Graph Attention\\
    on hex grid
};

\end{tikzpicture}
\end{document}
"""
    return tikz_code


def generate_simplified_tikz():
    """Generate a simplified horizontal flow diagram."""

    tikz_code = r"""
\documentclass[border=10pt, tikz]{standalone}
\usepackage{xcolor}

\definecolor{InputColor}{RGB}{158,158,158}
\definecolor{ConvColor}{RGB}{255,193,7}
\definecolor{HexColor}{RGB}{76,175,80}
\definecolor{FusionColor}{RGB}{33,150,243}
\definecolor{HeadColor}{RGB}{233,30,99}

\usetikzlibrary{positioning, shapes, arrows.meta, fit, backgrounds, calc}

\begin{document}
\begin{tikzpicture}[
    node distance=0.6cm and 1.2cm,
    every node/.style={font=\small},
    block/.style={rectangle, draw, rounded corners=3pt, minimum height=0.8cm, align=center},
    input/.style={block, fill=InputColor!20, minimum width=2cm},
    encoder/.style={block, minimum width=1.8cm, minimum height=0.6cm},
    conv/.style={encoder, fill=ConvColor!40},
    hex/.style={encoder, fill=HexColor!40},
    fusion/.style={block, fill=FusionColor!30, minimum width=2.5cm, minimum height=3cm},
    head/.style={block, fill=HeadColor!30, minimum width=1.5cm},
    arrow/.style={-{Stealth[length=2mm]}, thick, gray},
    brace/.style={decorate, decoration={brace, amplitude=5pt, raise=2pt}},
]

% Input
\node[input, minimum height=3cm] (input) {
    \textbf{Input}\\[2pt]
    4760 $\times$ 2\\[2pt]
    \tiny(sensors $\times$ channels)
};

% Face Encoders
\node[conv, right=1.5cm of input, yshift=1.2cm] (inner) {\footnotesize Inner 93$\times$44};
\node[conv, below=0.15cm of inner] (outer) {\footnotesize Outer 9$\times$24};
\node[conv, below=0.15cm of outer] (us) {\footnotesize US 24$\times$6};
\node[conv, below=0.15cm of us] (ds) {\footnotesize DS 24$\times$6};
\node[hex, below=0.15cm of ds] (top) {\footnotesize Top PMT};
\node[hex, below=0.15cm of top] (bot) {\footnotesize Bot PMT};

% Encoder label
\node[above=0.3cm of inner, font=\footnotesize\bfseries] {Face Encoders};

% Token notation
\node[right=0.8cm of us, font=\scriptsize, align=center] (tokens) {
    6 $\times$ 1024\\[-2pt]
    tokens
};

% Transformer
\node[fusion, right=2.5cm of us] (trans) {
    \textbf{Transformer}\\[4pt]
    \footnotesize 2 layers\\
    \footnotesize 8 heads\\[4pt]
    \footnotesize 6144-dim
};

% Task Heads
\node[head, right=1.5cm of trans, yshift=1cm] (angle) {\footnotesize Angle};
\node[head, below=0.2cm of angle] (energy) {\footnotesize Energy};
\node[head, below=0.2cm of energy] (timing) {\footnotesize Timing};
\node[head, below=0.2cm of timing] (pos) {\footnotesize Position};

% Head label
\node[above=0.3cm of angle, font=\footnotesize\bfseries] {Task Heads};

% Arrows from input
\foreach \enc in {inner, outer, us, ds, top, bot} {
    \draw[arrow] (input.east) -- ++(0.3,0) |- (\enc.west);
}

% Arrows to transformer
\foreach \enc in {inner, outer, us, ds, top, bot} {
    \draw[arrow] (\enc.east) -- (\enc.east -| trans.west);
}

% Arrows to heads
\foreach \h in {angle, energy, timing, pos} {
    \draw[arrow] (trans.east) -- ++(0.3,0) |- (\h.west);
}

% Legend
\node[conv, below=1.5cm of bot, minimum width=1cm, minimum height=0.4cm] (leg1) {};
\node[right=0.1cm of leg1, font=\scriptsize] {ConvNeXt V2};
\node[hex, right=1.5cm of leg1, minimum width=1cm, minimum height=0.4cm] (leg2) {};
\node[right=0.1cm of leg2, font=\scriptsize] {HexNeXt (GAT)};

% Title
\node[above=0.8cm of trans, font=\large\bfseries] {XEC Multi-Branch Model};

\end{tikzpicture}
\end{document}
"""
    return tikz_code


def generate_detailed_encoder_tikz():
    """Generate detailed view of ConvNeXt encoder block."""

    tikz_code = r"""
\documentclass[border=10pt, tikz]{standalone}
\usepackage{xcolor}

\definecolor{StemColor}{RGB}{255,235,59}
\definecolor{BlockColor}{RGB}{255,193,7}
\definecolor{DownColor}{RGB}{255,152,0}
\definecolor{PoolColor}{RGB}{121,85,72}

\usetikzlibrary{positioning, shapes, arrows.meta, fit}

\begin{document}
\begin{tikzpicture}[
    node distance=0.5cm,
    block/.style={rectangle, draw, rounded corners=2pt, minimum height=0.7cm, minimum width=2cm, align=center, font=\small},
    arrow/.style={-{Stealth[length=2mm]}, thick},
]

% Title
\node[font=\large\bfseries] (title) {ConvNeXt V2 Face Encoder};

% Stem
\node[block, fill=StemColor!50, below=0.5cm of title] (stem) {
    Stem\\[-2pt]
    \tiny Conv 4$\times$4, stride 4
};

% Block 1
\node[block, fill=BlockColor!50, below=0.4cm of stem] (block1) {
    2$\times$ ConvNeXt Block\\[-2pt]
    \tiny dim=32, GRN
};

% Downsample
\node[block, fill=DownColor!50, below=0.4cm of block1] (down) {
    Downsample\\[-2pt]
    \tiny LayerNorm + Conv 2$\times$2
};

% Block 2
\node[block, fill=BlockColor!50, below=0.4cm of down] (block2) {
    3$\times$ ConvNeXt Block\\[-2pt]
    \tiny dim=64, GRN
};

% Global Pool
\node[block, fill=PoolColor!30, below=0.4cm of block2] (pool) {
    Global Avg Pool\\[-2pt]
    \tiny $\rightarrow$ 1024-dim token
};

% Arrows
\draw[arrow] (stem) -- (block1);
\draw[arrow] (block1) -- (down);
\draw[arrow] (down) -- (block2);
\draw[arrow] (block2) -- (pool);

% Input/Output labels
\node[left=0.3cm of stem, font=\scriptsize] {Input: H$\times$W$\times$2};
\node[left=0.3cm of pool, font=\scriptsize] {Output: 1024};

\end{tikzpicture}
\end{document}
"""
    return tikz_code


if __name__ == "__main__":
    # Create output directory
    os.makedirs(".", exist_ok=True)

    # Generate main architecture diagram
    with open("xec_architecture.tex", "w") as f:
        f.write(generate_xec_tikz())
    print("Generated: xec_architecture.tex")

    # Generate simplified diagram
    with open("xec_simplified.tex", "w") as f:
        f.write(generate_simplified_tikz())
    print("Generated: xec_simplified.tex")

    # Generate encoder detail
    with open("xec_encoder_detail.tex", "w") as f:
        f.write(generate_detailed_encoder_tikz())
    print("Generated: xec_encoder_detail.tex")

    print("\nTo compile:")
    print("  pdflatex xec_architecture.tex")
    print("  pdflatex xec_simplified.tex")
    print("  pdflatex xec_encoder_detail.tex")
