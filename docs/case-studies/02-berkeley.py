# %% [markdown]
# # Berkeley, 1973: the admissions paradox that wasn't
#
# Fall 1973. The University of California, Berkeley admits 44 percent of the men who apply to
# graduate school and 30 percent of the women. Fourteen points. The kind of gap that ends up
# in a lawsuit, and nearly did. On its face it says the university sorted applicants by sex
# and let more men through.
#
# It didn't. Peter Bickel, Eugene Hammel and J. William O'Connell took the admissions records
# apart department by department and published the result in Science in 1975. The pooled gap
# is real arithmetic, and the conclusion drawn from it is wrong, and the distance between
# those two facts is the whole subject of causal inference. Guide one built this same reversal
# in a simulated loyalty program and named it Simpson's paradox. Here it is on the data that
# made it famous.
#
# The library ships the table, so you can check every number yourself.

# %%
import numpy as np
import pandas as pd

from ergodic.datasets import load_berkeley

ds = load_berkeley()
df = ds.data.to_pandas()
print(f"{len(df)} applicants")
df.head()

# %% [markdown]
# One row per applicant: their sex, the department they applied to (the six largest, called A
# through F), and whether they got in. Start with the number that started the trouble, the
# admission rate for each sex pooled over all six departments.

# %%
pooled = df.groupby("gender")["admit"].mean()
print((pooled * 100).round(1))
print(f"\npooled gap (women minus men): {(pooled['Female'] - pooled['Male']) * 100:+.1f} points")

# %% [markdown]
# Men 44.5 percent, women 30.4 percent. If that were the end of the data, the case would be
# hard to argue with. So go one level down, the level the headline skips, and read the
# admission rate for each sex inside each department.

# %%
within = df.groupby(["dept", "gender"])["admit"].mean().unstack() * 100
within["gap"] = within["Female"] - within["Male"]
within.round(1)

# %% [markdown]
# Look at the gap column. In department A women are admitted at 82 percent against 62 for men,
# a twenty-point edge to women. B, D and F lean the same way. Only C and E tilt toward men,
# and by a few points. The sex that loses by fourteen points in the pooled table wins, or
# ties, in four of the six departments where the decisions get made. Both statements
# come from the same 4526 rows. Neither is a mistake.
#
# So where does the pooled gap come from? Not from the admission rates. From who applied
# where. Count the applications.

# %%
applied = df.groupby(["dept", "gender"]).size().unstack()
share = (applied / applied.sum() * 100).round(1)
share.columns = ["Female %", "Male %"]
pd.concat([applied, share], axis=1)

# %% [markdown]
# Read down the two share columns. Men pile into A and B, which admit roughly two applicants
# in three. Almost nobody is turned away there. Women apply to C, D, E and F, where the
# admission rate runs from 35 percent down to 6. The departments women chose are harder to get
# into, for everyone. The pooled rate for women is low because it is an average over
# the competitive departments, weighted by where women applied, and that weighting is the
# entire effect. Department is the confounder. The application pattern is how it does its
# work.
#
# A bar per department makes the split visible. Within each one the two bars sit close; the
# departments on the right are short for both sexes.

# %%
import matplotlib.pyplot as plt


def draw_within(table):
    depts = list(table.index)
    x = np.arange(len(depts))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x - 0.19, table["Female"], width=0.36, color="#d1495b", label="women")
    ax.bar(x + 0.19, table["Male"], width=0.36, color="#33415c", label="men")
    ax.set_xticks(x)
    ax.set_xticklabels(depts)
    ax.set_xlabel("department")
    ax.set_ylabel("admitted (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    fig.tight_layout()


draw_within(within)

# %% [markdown]
# ## The naive answer
#
# Write the pooled gap as a causal claim and you get the simplest graph there is: sex points
# at admission, nothing else in the picture. Ask the library to estimate it.

# %%
from ergodic import dag, estimate_effect

naive = estimate_effect(dag(["gender -> admit"]), ds.data, "gender", "admit", treated="Female")
print(naive)
print(f"\nnaive effect of being female on admission: {naive.value:+.3f}")

# %% [markdown]
# About minus fourteen points, the headline in risk-difference form. It tempts because it is
# the number you can compute without thinking, straight off the pooled table, and it sounds
# like the answer to the question everyone is asking. It is the answer to a different
# question: what is the admission rate of women versus men as they sorted themselves across
# departments. That is a fact about application behaviour, not about how the
# committees treated an application.
#
# ## Why it is wrong
#
# Department sits upstream of both things that matter. It shapes where a woman applies, and it
# sets how hard admission is once she does. That is a back door: a path from sex to admission
# that runs through department and carries association without any causation along it. The
# pooled gap adds the back door to whatever the committees did and reports the sum. Guide one
# drew the same fork over tenure, members and spend. Same shape, same trap.
#
# ## Close the door
#
# Put department in the graph as the common cause it is. Sex and admission each get an arrow
# out of it, and sex keeps its arrow into admission, the quantity we want.

# %%
g = dag(["dept -> gender", "dept -> admit", "gender -> admit"])

# %%
from matplotlib.patches import FancyArrowPatch


def draw_dag(graph, pos):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for edge in graph.edges:
        u, v = edge.directed_pair()
        ax.add_patch(
            FancyArrowPatch(
                pos[u],
                pos[v],
                arrowstyle="-|>",
                mutation_scale=18,
                shrinkA=30,
                shrinkB=30,
                lw=1.6,
                color="#33415c",
            )
        )
    for name, (x, y) in pos.items():
        ax.text(
            x,
            y,
            name,
            ha="center",
            va="center",
            fontsize=11,
            bbox={"boxstyle": "circle,pad=0.45", "fc": "#eef1f6", "ec": "#33415c"},
        )
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(-0.2, 1.45)
    ax.axis("off")
    fig.tight_layout()


draw_dag(g, pos={"dept": (1.0, 1.2), "gender": (0.0, 0.0), "admit": (2.0, 0.0)})

# %% [markdown]
# Ask the graph what to adjust for. It reads the back door and hands back the set that closes
# it, with the formula to apply.

# %%
from ergodic import identify_effect

ident = identify_effect(g, "gender", "admit")
print("adjustment set:", set(ident.adjustment_set))
print("estimand:      ", ident.estimand)

# %% [markdown]
# Department, and only department. Now estimate the effect through that formula, the same one
# call as before but on the honest graph.

# %%
adjusted = estimate_effect(g, ds.data, "gender", "admit", treated="Female")
print(adjusted)
print(f"\ndepartment-adjusted effect: {adjusted.value:+.3f}")

# %% [markdown]
# The sign flips. Compare like with like, women against men inside the same department, and
# being a woman carries a small positive effect on admission, about four points, with an
# interval that clears zero. The fourteen-point penalty was the departments talking, not the
# committees. Side by side:

# %%
summary = pd.DataFrame(
    {
        "effect": [naive.value, adjusted.value],
        "low": [naive.interval()[0], adjusted.interval()[0]],
        "high": [naive.interval()[1], adjusted.interval()[1]],
    },
    index=["pooled (gender -> admit)", "adjusted for department"],
)
summary.round(3)

# %% [markdown]
# Two estimates off the same 4526 rows, opposite signs, and the only thing that changed is one
# arrow in the graph. The data never picked between them. The graph did.
#
# ## What the lesson is, and what it is not
#
# Read this as condition on the pre-treatment common cause, and stop there. The fix here was
# department, a thing fixed before any committee saw a file. It is not slice by everything you
# have. Guide two builds three worlds where the extra control is a mediator or a collider, and
# adjusting for it moves the answer away from the truth instead of toward it. Department earns
# its place in the adjustment set because of where it sits in the graph, not because it was
# lying around in the dataframe.
#
# One cheap check before you trust the reversal. Scramble the sex label so it points at
# nothing, then re-run the whole adjusted pipeline. A real effect should collapse to zero
# under that shuffle; if it doesn't, the estimate is reading something other than sex.

# %%
from ergodic import refute_effect

ref = refute_effect(
    g,
    ds.data,
    "gender",
    "admit",
    method="placebo_treatment",
    treated="Female",
    n_simulations=20,
    random_state=0,
)
print(ref)
print(f"\nreal estimate:     {ref.estimand_value:+.3f}")
print(f"scrambled label:   {ref.refuted_value:+.3f}")

# %% [markdown]
# The placebo effect lands near zero, a small fraction of the real one, and the permutation
# p-value says a four-point effect rarely shows up by chance once the link to sex is cut. The
# check passes. It doesn't prove the graph is right, nothing on observational data can, but a
# failure here would have been a real warning and there isn't one.
#
# Berkeley is the famous case, and it isn't the only one. A 1986 study in the British Medical
# Journal compared two treatments for kidney stones: the better treatment lost in the overall
# table and won for small stones and for large stones separately, because surgeons sent it the
# harder cases. Guide one cites it for the same reason. Same paradox, surgery instead of
# admissions, a life on the line instead of a graduate seat.
#
# If you want the mechanism from scratch, on a world where the true effect is written into the
# code so you can grade the answer, start at [why causality](../learn/01-why-causality.py).
