import raw_operator as O
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    d=json.loads((O.ROOT/'ode_simple_constants.json').read_text());np=O.np
    fig,axs=plt.subplots(1,2,figsize=(10,4.1))
    t=np.linspace(0,4*np.pi,200);u=np.sin(t)+1.3*np.cos(t);v=np.cos(t)-1.3*np.sin(t)
    axs[0].plot(u,v,color='#2176ae');axs[0].set_aspect('equal');axs[0].set_xlabel('u');axs[0].set_ylabel("u'")
    axs[0].set_title("ODE_simple: u² + (u')² = 2.69");axs[0].grid(alpha=.2)
    for j,name in enumerate(['energy','phase_sin','phase_cos']):
        for offset,tag,label,col in [(-.12,name,'With constant','#2176ae'),(.12,name+'_without_constant','Without constant','#ce6c34')]:
            rr=[r for r in d['summary'] if r['noise']==5 and r['candidate']==tag and r['method']=='continuous_clsm_128']
            vals=[r['clean_energy'] for r in rr]
            axs[1].scatter(np.full(3,j+offset),vals,color=col,label=label if j==0 else None)
            axs[1].plot([j+offset-.07,j+offset+.07],[np.median(vals)]*2,color=col)
    axs[1].set_yscale('log');axs[1].set_xticks(range(3),['Energy','Phase sin','Phase cos'])
    axs[1].set_ylabel('Clean residual energy / target energy');axs[1].set_title('5% noise; three seeds; fixed-panel weak fits')
    axs[1].grid(axis='y',alpha=.2);axs[1].legend();fig.tight_layout()
    fig.savefig(O.ROOT/'ode_simple_constants.png',dpi=180);fig.savefig(O.ROOT/'ode_simple_constants.svg');plt.close(fig)


if __name__=='__main__':main()
