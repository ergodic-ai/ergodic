# %% [markdown]
# # Discovery on a known network: grading the graph
#
# Every other case study handed you the graph. Someone had drawn the arrows, and the only
# job left was to read an effect off them. That is a fair description of the easy half of
# causal work. The hard half is the question it skips: where does the graph come from when
# nobody can draw it?
#
# Discovery learns the structure from the data. You feed it a table and it returns the arrows
# it can defend, with the undecided ones left marked rather than guessed. The trouble with
# trying it on real data is that you can never grade the answer. Nobody knows the true graph
# of a real business, so a discovered graph is a claim you have to take partly on faith.
#
# The fix is the same move from [why causality](../learn/01-why-causality.py): simulate a
# world whose truth you wrote yourself, then grade the method against the answer key. Here
# the world is a textbook Bayesian network. The Asia chest-clinic network (Lauritzen and
# Spiegelhalter, 1988) has eight variables and eight known edges, so when discovery returns a
# graph we can score it edge by edge. That is the whole point of running it here. The number
# you get is not a guess about a method that might work; it is a measurement of one that did.

# %%
from ergodic.datasets import load_asia

ds = load_asia()  # n=3000 by default; the draw is seeded and deterministic
print(ds.target["source"])
print("nodes:", ds.target["nodes"])
print("true edges:", ds.target["n_edges"])
ds.data

# %% [markdown]
# Eight binary variables, one row per simulated patient. A recent visit to Asia, tuberculosis,
# whether they smoke, lung cancer, bronchitis, a logical "either tuberculosis or lung cancer",
# a chest x-ray, and dyspnoea (shortness of breath). The true graph wires them into a small
# clinic: smoking drives lung cancer and bronchitis, a visit to Asia raises the tuberculosis
# risk, tuberculosis and lung cancer feed the `either` node, and `either` plus bronchitis
# explain the x-ray and the breathing trouble.
#
# Before discovery, look at the marginals. How common is each condition?

# %%
df = ds.data.to_pandas()
rates = (df == "yes").mean().sort_values()
print(rates.round(4))

# %% [markdown]
# Smoking is a coin flip and bronchitis is close behind, but the diseases are rare. A visit to
# Asia sits near 1 percent, and tuberculosis is rarer still. Hold onto that. A cause that fires
# in one row out of a hundred leaves a faint trace in the data, and a faint trace is hard for
# any test to separate from noise. The arrow `asia -> tub` connects the two rarest variables in
# the table, so if discovery is going to miss an edge, that is the one to bet on.
#
# ## Run discovery and grade it
#
# The truth is a DAG built straight from the known edges. The estimate comes from one call to
# `discover` with the PC algorithm, which is the constraint-based method the
# [discovery guide](../learn/06-discovering-structure.py) walks through. It returns a CPDAG:
# arrows where the data decides direction, dashes where several directions explain the data
# equally well.

# %%
from ergodic import dag, discover

truth = dag(list(ds.target["edges"]))
result = discover(ds.data, method="pc")
print(result.graph.kind.value)
sorted(e.glyph() for e in result.graph.edges)

# %% [markdown]
# To grade it, compare skeletons. The skeleton is the graph with directions thrown away, just
# the set of pairs that share an edge. Scoring the skeleton asks the first question that
# matters: did discovery find the right connections at all? Direction is a separate question,
# and on a CPDAG some directions are genuinely undecidable, so skeleton recall is the honest
# headline.


# %%
def skeleton(graph):
    return {frozenset((x, y)) for x in graph.nodes for y in graph.neighbors(x)}


truth_skeleton = skeleton(truth)
found_skeleton = skeleton(result.graph)

recall = len(truth_skeleton & found_skeleton) / len(truth_skeleton)
precision = len(truth_skeleton & found_skeleton) / max(1, len(found_skeleton))
print(f"recall    {recall:.3f}   (found edges that are real)")
print(f"precision {precision:.3f}   (real edges among those found)")

# %% [markdown]
# Recall is 0.625 and precision is 1.0. Read those two numbers together. Precision 1.0 says
# every edge PC drew is a true edge: it invented nothing, added no phantom link between
# variables that only correlate. That is the trait you want from a discovery method on data you
# cannot check, because a false edge is a confounder you will adjust for that was never there.
# Recall 0.625 says it found five of the eight real edges and left three on the table.
#
# Which three? Name them, because the misses are not random.

# %%
missed = truth_skeleton - found_skeleton
for pair in sorted(tuple(sorted(p)) for p in missed):
    print(" - ".join(pair))

# %% [markdown]
# `asia - tub`, `either - xray`, and `either - dysp`. Two of the three touch `either`, and the
# third is the rare-cause edge we flagged from the marginals. Both misses have an honest reason.
#
# The `asia -> tub` edge links the two rarest variables in the table. Tuberculosis appears in
# under 1 percent of patients and a visit to Asia in about 1 percent, so the slice where Asia
# moves the tuberculosis rate is a handful of rows in three thousand. The signal is
# real but tiny, and PC's independence test cannot yet tell it from sampling noise. This is not
# a flaw in the algorithm. It is the data being honest that it has not seen enough.
#
# The `either` misses come from how that node is built. In the network `either` is a logical or:
# it is "yes" exactly when tuberculosis or lung cancer is present, with no randomness of its own.
# A deterministic node is a strange object for a conditional-independence test. Once you know
# tuberculosis and lung cancer, you know `either` perfectly, which scrambles the independence
# pattern its children depend on, and the edges from `either` to the x-ray and to dyspnoea get
# lost. Determinism breaks an assumption these tests quietly rely on, so the honest result is a
# gap, not a confident wrong arrow.
#
# Here is the recovered graph. The five edges PC kept, drawn over the clinic.

# %%
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

POS = {
    "asia": (0.0, 2.0),
    "tub": (0.0, 1.0),
    "smoke": (3.0, 2.0),
    "lung": (2.4, 1.0),
    "bronc": (3.6, 1.0),
    "either": (1.2, 1.0),
    "xray": (0.6, 0.0),
    "dysp": (2.4, 0.0),
}


def draw_graph(graph, pos, title, missed_pairs=()):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for a, b in missed_pairs:
        ax.plot(
            [pos[a][0], pos[b][0]],
            [pos[a][1], pos[b][1]],
            color="#d1495b",
            ls=":",
            lw=1.3,
            zorder=1,
        )
    for edge in graph.edges:
        u, mid, v = edge.glyph().split()
        if mid == "<--":
            u, v, mid = v, u, "-->"
        style = "-" if mid == "---" else "-|>"
        ax.add_patch(
            FancyArrowPatch(
                pos[u],
                pos[v],
                arrowstyle=style,
                mutation_scale=16,
                shrinkA=30,
                shrinkB=30,
                lw=1.6,
                color="#33415c",
                zorder=2,
            )
        )
    for name, (x, y) in pos.items():
        ax.text(
            x,
            y,
            name,
            ha="center",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.4", "fc": "#eef1f6", "ec": "#33415c"},
            zorder=3,
        )
    ax.set_xlim(-0.6, 4.2)
    ax.set_ylim(-0.6, 2.6)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()


draw_graph(
    result.graph,
    POS,
    "Asia at n=3000: five solid edges, three missed (dotted red)",
    missed_pairs=[tuple(sorted(p)) for p in missed],
)

# %% [markdown]
# The funnel that survives is the spine of the clinic. Smoking, lung cancer, bronchitis, and the
# two `either` parents all sit where they should. The dotted red lines are the three edges
# discovery could not yet justify, and every one of them runs into a rare variable or the
# deterministic node. The graph is partial, and it is partial in exactly the places the data is
# thin.
#
# ## More data sharpens the picture
#
# The rare-cause miss was a sample-size problem, so give the algorithm more samples. The loader
# takes a row count, and nothing else about the world changes.

# %%
ds_big = load_asia(n=8000)
result_big = discover(ds_big.data, method="pc")

found_big = skeleton(result_big.graph)
recall_big = len(truth_skeleton & found_big) / len(truth_skeleton)
precision_big = len(truth_skeleton & found_big) / max(1, len(found_big))
print(f"n=8000   recall {recall_big:.3f}   precision {precision_big:.3f}")

missed_big = truth_skeleton - found_big
print("still missed:", sorted(" - ".join(sorted(p)) for p in missed_big))

# %% [markdown]
# Recall climbs from 0.625 to 0.75, and precision stays pinned at 1.0. The edge that came back
# is `asia -> tub`: with 8000 rows instead of 3000, the faint trace of a rare cause finally
# clears the test's bar, and PC draws it. No knob was turned and no hint was added. The same
# method, fed more of the same data, recovered an edge it had honestly declined before.
#
# What is still missing are the two `either` edges, and they stay missing because their cause is
# structural, not statistical. More rows do not undo determinism. That distinction is the useful
# lesson here. A miss from thin data heals with sample size; a miss from a broken assumption does
# not, and it tells you something real about the world rather than the dataset.

# %%
draw_graph(result_big.graph, POS, "Asia at n=8000: asia -> tub returns")

# %% [markdown]
# ## A clean network for contrast
#
# Asia recovers partly because Asia is hard: rare diseases and a logical gate. A simpler network
# shows what discovery does when the world cooperates. The burglar-alarm network (Pearl, 1988)
# has five variables and four edges. A burglary or an earthquake can trip an alarm, and the
# alarm is what makes a neighbor call.

# %%
from ergodic.datasets import load_earthquake

eq = load_earthquake()
print("nodes:", eq.target["nodes"])
print("true edges:", eq.target["edges"])

res = discover(eq.data, method="pc")
eq_truth = skeleton(dag(list(eq.target["edges"])))
eq_found = skeleton(res.graph)
print(f"recall {len(eq_truth & eq_found) / len(eq_truth):.3f}")
sorted(e.glyph() for e in res.graph.edges)

# %% [markdown]
# Full skeleton, recall 1.0, and the arrows are oriented too. The interesting one is the alarm.
# Both a burglary and an earthquake point into it, and that meeting of two arrowheads at one
# node is a collider, the single structure observational data can orient on its own. Burglaries
# and earthquakes are independent until the alarm goes off; learning the alarm fired makes them
# dependent, because one cause explains away the other. That signature is what fixes the arrows,
# and you can read it straight off the recovered graph.

# %%
from ergodic import Mark

at_alarm_b = res.graph.mark_at("Burglary", "Alarm", "Alarm")
at_alarm_e = res.graph.mark_at("Earthquake", "Alarm", "Alarm")
print("Burglary -> Alarm, mark at Alarm:", at_alarm_b)
print("Earthquake -> Alarm, mark at Alarm:", at_alarm_e)
print("both arrowheads into Alarm:", at_alarm_b is Mark.ARROW and at_alarm_e is Mark.ARROW)

# %% [markdown]
# Both endpoints at the alarm are arrowheads, so PC recovered the v-structure exactly. The
# contrast with Asia is the message. A small clean network with strong signals recovers in full,
# while a harder one recovers in part and is plain about which part. Same algorithm, two
# honesties, and neither of them pretends to know more than the data supports.
#
# ## Carrying this back to a real table
#
# On real data you never get an answer key, so you never get a recall number either. What
# carries over is the behavior these known networks just showed you. PC adds no edges it can't
# support, it recovers the structure where the signal is strong, and it leaves gaps where a
# cause is rare or an assumption is strained. A partial graph that's honest about its gaps
# beats a complete graph that guessed.
#
# A discovered graph is not the end of the line. It feeds straight into the rest of the library:
# `identify_effect` reads a CPDAG and `estimate_effect` runs off it, so the arrows you just
# learned become the effects you wanted, with the bookkeeping shown in the
# [discovery guide](../learn/06-discovering-structure.py).
