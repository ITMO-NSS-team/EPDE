"""Forecasts and fresh margins for the matched two-system comparison."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

root=Path(__file__).resolve().parent
ac=root.parent/"ac_diffusion_uncertainty_2026-11-18"
fig,axes=plt.subplots(1,3,figsize=(13.8,4.0),sharey=True,layout="constrained")
rng=np.random.default_rng(445)
for ax,directory,title,shifted in zip(axes,(ac,root,root),
        ("Allen–Cahn","Navier–Stokes","Navier–Stokes, recentered"),(False,False,True)):
    with (directory/"forecast_check.csv").open() as f:
        forecasts=list(csv.DictReader(f))
    if shifted:
        ranges=json.loads((directory/"confirmation/frozen.json").read_text())["ranges"]
        for r,s in zip(forecasts,ranges):
            assert int(r["anchor"])==s["anchor"]
            r.update(lower=s["shifted_lower"],upper=s["shifted_upper"],bootstrap_median=r["observed"])
        with (directory/"confirmation/margins.csv").open() as f:
            fresh=np.array([float(r["refitted"]) for r in csv.DictReader(f)])
    else:
        with (directory/"replicates.csv").open() as f:
            fresh=np.array([float(r["refitted"]) for r in csv.DictReader(f) if r["cohort"]=="fresh"])
    ax.axvline(0,ls="--",color="#333333",lw=1)
    for j,r in enumerate(forecasts):
        lo,hi,median,observed=[1e4*float(r[k]) for k in ("lower","upper","bootstrap_median","observed")]
        ax.plot([lo,hi],[j,j],color="#2c6489",lw=2.7,solid_capstyle="butt")
        ax.plot([lo,hi],[j,j],"|",ms=7,color="#2c6489")
        ax.plot(median,j,"o",ms=4,color="#2c6489")
        ax.plot(observed,j,"D",ms=4,mfc="white",mec="#9d430d")
    ax.scatter(1e4*fresh,6.3+rng.uniform(-.19,.19,len(fresh)),s=9,alpha=.48,color="#24614a",linewidths=0)
    ax.set_title(title,loc="left")
    ax.grid(axis="x",color="#e2e7e9",linewidth=.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y",length=0)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.set_yticks([*range(6),6.3],[r["anchor"] for r in forecasts]+["Fresh fields (128)"])
axes[0].set_ylim(6.9,-.55)
axes[0].set_ylabel("Observation seed")
# Identical score units and limits preserve the cross-system difference in scale.
limits=[ax.get_xlim() for ax in axes]
for ax in axes:
    ax.set_xlim(min(x[0] for x in limits),max(x[1] for x in limits))
fig.supxlabel("Score difference (reduced − full), in units of 10⁻⁴\nPositive favors retaining the tested term",fontsize=10)
axes[0].legend(handles=[
    Line2D([],[],color="#2c6489",lw=2.7,marker="o",ms=4,label="Nominal 95% repeat range and median"),
    Line2D([],[],ls="none",marker="D",ms=4,mec="#9d430d",mfc="white",label="Observed difference")],
    loc="lower left",bbox_to_anchor=(0,1.1),frameon=False,ncol=2,fontsize=9)
fig.savefig(root/"two_system_ranges.png",dpi=180)
plt.close(fig)
