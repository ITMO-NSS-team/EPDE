"""Repeat-measurement ranges and the independent distribution on a common axis."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

root=Path(__file__).resolve().parent
with (root/"forecast_check.csv").open() as f:
    forecasts=list(csv.DictReader(f))
with (root/"replicates.csv").open() as f:
    fresh=np.array([float(r["refitted"]) for r in csv.DictReader(f) if r["cohort"]=="fresh"])*1.e5
fig,ax=plt.subplots(figsize=(8.4,4.6),layout="constrained")
ax.axvline(0,color="#30343b",linestyle="--",linewidth=1,zorder=0)
color="#285e85"
for j,r in enumerate(forecasts):
    lo,hi,obs,med=np.array([float(r[k]) for k in ("lower","upper","observed","bootstrap_median")])*1.e5
    ax.plot([lo,hi],[j,j],color=color,linewidth=3,solid_capstyle="butt")
    ax.plot([lo,hi],[j,j],"|",color=color,markersize=8)
    ax.plot(med,j,"o",color=color,markersize=5)
    ax.plot(obs,j,"D",markeredgecolor="#aa4818",markerfacecolor="white",markersize=5,markeredgewidth=1.3)
rng=np.random.default_rng(812)
y=len(forecasts)+.4
ax.scatter(fresh,y+rng.uniform(-.22,.22,len(fresh)),s=12,alpha=.6,color="#294e42",linewidths=0)
ax.set_yticks([*range(len(forecasts)),y],[r["anchor"] for r in forecasts]+["Fresh fields (128)"])
ax.set_ylim(y+.65,-.62)
ax.set_xlabel("Score difference (reaction only − full), in units of 10⁻⁵\nPositive values favor retaining diffusion")
ax.set_ylabel("Observation seed")
ax.set_title("Allen–Cahn diffusion on repeated noisy measurements",loc="left",fontsize=13)
ax.spines[["top","right","left"]].set_visible(False)
ax.tick_params(axis="y",length=0)
ax.grid(axis="x",color="#e2e5e8",linewidth=.6)
ax.set_axisbelow(True)
ax.legend(handles=[
    Line2D([],[],color=color,lw=3,marker="o",markersize=5,label="95% repeat range and median"),
    Line2D([],[],linestyle="none",marker="D",markeredgecolor="#aa4818",markerfacecolor="white",
           markersize=5,label="Observed difference")],loc="lower left",bbox_to_anchor=(0,1.07),
    frameon=False,ncol=2,fontsize=9)
fig.savefig(root/"repeat_ranges.png",dpi=170)
plt.close(fig)
