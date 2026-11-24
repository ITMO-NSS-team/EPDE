from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
sp=json.loads((ROOT/"confirmation"/"specificity.json").read_text())
read=json.loads((ROOT/"readout_summary.json").read_text())["metrics"]
fig,axes=plt.subplots(1,2,figsize=(11.4,4.4))
colors=("#18598a","#bd572b")
plots=[
    (["Original fit","Spatial-mean fit","Disjoint spatial-mean fit","Fit with modes 0–12"],
     [(sp[k]["positive_wins"],sp[k]["false_wins"]) for k in ("local","mean_staged","disjoint_mean","high_only")],
     64,"Original held-out score","Positive score margin"),
    (["Modes 0–3","Modes 4–12","Modes 0–12","Fitted |diffusion coefficient|"],
     [(read[k]["detected"],read[k]["false_calls"]) for k in ("low","high","all","abs_diffusion")],
     128,"Calibrated Fourier checks","Threshold from 64 separate zero-diffusion fields"),
]
for ax,(labels,values,total,title,detail) in zip(axes,plots):
    y=np.arange(len(labels))
    for i,(color,legend) in enumerate(zip(colors,("Diffusion present","Diffusion absent"))):
        counts=np.array([v[i] for v in values])
        positions=y+(i-.5)*.29
        ax.barh(positions,counts,.27,color=color,label=legend)
        for yp,n in zip(positions,counts):
            ax.text(n+(.8 if total==64 else .35),yp,str(n),va="center",fontsize=9,color=color)
    ax.set_yticks(y,labels)
    ax.invert_yaxis()
    ax.set_xlim(0,66 if total==64 else 20)
    ax.set_xticks([0,16,32,48,64] if total==64 else [0,5,10,15,20])
    ax.set_xlabel(f"Fields favoring diffusion (out of {total})")
    ax.set_title(title,loc="left",fontsize=13,fontweight="bold",pad=25)
    ax.text(0,1.02,detail,transform=ax.transAxes,fontsize=9,color="#414952")
    ax.spines[["right","top","left"]].set_visible(False)
    ax.tick_params(axis="y",length=0)
    ax.grid(axis="x",alpha=.16)
    ax.set_axisbelow(True)
axes[0].legend(loc="upper left",bbox_to_anchor=(-.02,-.22),ncol=2,frameon=False)
fig.subplots_adjust(left=.19,right=.98,wspace=.98,top=.82,bottom=.24)
fig.savefig(ROOT/"diffusion_comparison.png",dpi=190,bbox_inches="tight",facecolor="white")
