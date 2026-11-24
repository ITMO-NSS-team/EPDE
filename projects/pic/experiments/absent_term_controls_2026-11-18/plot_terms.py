"""Allen–Cahn sign frequencies and recentered decisions for every tested term."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
summary=json.loads((root/"summary.json").read_text())
with (root/"per_anchor.csv").open(encoding="utf-8") as stream:
    forecasts=list(csv.DictReader(stream))
terms=summary["outcomes"]
fig,(freq,heat)=plt.subplots(1,2,figsize=(8.1,4.8),gridspec_kw={"width_ratios":[1.1,1]},layout="constrained")
labels={"u_xx":"uₓₓ"}
colors={"retain":"#23785a","drop":"#af5a38","unresolved":"#e2e4e7"}
symbols={"retain":"+","drop":"−","unresolved":"?"}
anchors=sorted({r["anchor"] for r in forecasts})
for i,row in enumerate(terms):
    p=row["wins_retain"]/row["total"]
    lo,hi=row["p_retain_ci"]
    freq.hlines(i,lo,hi,color="#2d6185",lw=2)
    freq.plot(p,i,"o",ms=5,color="#2d6185")
    for j,anchor in enumerate(anchors):
        r=next(r for r in forecasts if r["key"]==row["key"] and r["anchor"]==anchor)
        dec=r["shifted_decision"]
        heat.add_patch(plt.Rectangle((j-.49,i-.48),.98,.96,facecolor=colors[dec],edgecolor="white"))
        heat.text(j,i,symbols[dec],ha="center",va="center",color="#30333a" if dec=="unresolved" else "white",fontsize=11)
freq.axvline(.5,ls="--",lw=1,color="#666666")
freq.set_xlim(-.04,1.04)
freq.set_xticks([0,.25,.5,.75,1],["0","25","50","75","100"])
freq.set_xlabel("Fresh fields favoring retention (%)")
freq.set_yticks(range(len(terms)),[labels.get(r["term"],r["term"]) for r in terms])
heat.set_yticks([])
heat.set_xticks(range(len(anchors)),anchors,rotation=35,ha="right",fontsize=8)
heat.set_xlim(-.5,len(anchors)-.5)
heat.set_xlabel("Observation seed")
freq.set_title("128 fresh fields",loc="left",fontsize=12)
heat.set_title("Recentered range by observation",loc="left",fontsize=12)
for ax in (freq,heat):
    ax.set_ylim(len(terms)-.5,-.5)
    ax.axhline(2.5,color="#42464b",lw=.8)
    ax.tick_params(length=0)
    ax.spines[["top","left","right"]].set_visible(False)
fig.suptitle("Allen–Cahn: three physical terms and six benchmark-absent additions",fontsize=12)
fig.supxlabel("Bars are exact 95% binomial intervals.    + retain     − drop     ? unresolved",fontsize=9)
fig.savefig(root/"term_comparisons.png",dpi=180)
plt.close(fig)
