"""Publication figures from frozen numerical tables; no synthetic observations."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'generated'/'figures';OUT.mkdir(parents=True,exist_ok=True)
COLORS={'M56':'#245B7B','M61':'#BD6630'}
SETS={'coverall':'Coverall recording','everyday':'Everyday workwear recording'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':250})
def read(name):return pd.read_csv(ROOT/'generated'/'results'/(name+'.csv'),keep_default_na=False)
def main():
    performance=read('performance');performance=performance[performance.reference=='both']
    fig,axes=plt.subplots(2,2,figsize=(10.5,6.8),sharey=True)
    for i,dataset in enumerate(SETS):
        for j,model in enumerate(COLORS):
            ax=axes[i,j];d=performance[(performance.dataset==dataset)&(performance.model==model)].sort_values('combination');x=np.arange(1,9)
            exact=d.exact_pct.to_numpy(float);wrong=(d.complete_pct-d.exact_pct).to_numpy(float);na=100-d.complete_pct.to_numpy(float)
            ax.bar(x,exact,color=COLORS[model],label='Exact complete code')
            ax.bar(x,wrong,bottom=exact,color='#B6C1C9',label='Complete mismatch')
            ax.bar(x,na,bottom=exact+wrong,color='#F4F4F4',edgecolor='#A7A7A7',linewidth=.5,hatch='//',label='Unresolved code')
            for a,v,n in zip(x,exact,d.exact_n):ax.text(a,v+.9,str(n),ha='center',va='bottom',fontsize=8.5)
            ax.set_title(f'{SETS[dataset]}  {model}');ax.set_xticks(x);ax.set_ylim(0,108);ax.set_yticks([0,25,50,75,100]);ax.set_xlabel('Information condition')
            if j==0:ax.set_ylabel('Frames (%)')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.005));fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(OUT/'figure2_original_outcomes.png',bbox_inches='tight');plt.close(fig)

    components=read('component_performance');components=components[components.reference=='both']
    fig,axes=plt.subplots(2,2,figsize=(10.2,7.6),sharex=True,sharey=True)
    for i,dataset in enumerate(SETS):
        for j,model in enumerate(COLORS):
            ax=axes[i,j];arr=np.array([[components[(components.dataset==dataset)&(components.model==model)&(components.combination==c)&(components.component==co)].iloc[0].exact_pct for co in ['back','arms','legs']] for c in range(1,9)],float)
            ax.imshow(arr,vmin=0,vmax=100,cmap='Blues',aspect='auto')
            for rr in range(8):
                for cc in range(3):ax.text(cc,rr,f'{arr[rr,cc]:.1f}',ha='center',va='center',fontsize=10,color='white' if arr[rr,cc]>57 else 'black')
            ax.set_title(f'{SETS[dataset]}  {model}');ax.set_xticks([0,1,2],['Back','Arms','Legs']);ax.tick_params(labelbottom=True);ax.set_yticks(range(8),[f'C{c}' for c in range(1,9)]);ax.spines[['left','bottom']].set_visible(False)
    fig.subplots_adjust(left=.06,right=.89,bottom=.07,top=.94,hspace=.23,wspace=.20)
    cax=fig.add_axes([.925,.13,.018,.72]);fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,100),cmap='Blues'),cax=cax,label='Matches / all frames (%)')
    fig.savefig(OUT/'figure3_component_agreement.png',bbox_inches='tight');plt.close(fig)

    contrasts=read('ordered_bootstrap_contrasts');d=contrasts[(contrasts.contrast_type=='information')&(contrasts.endpoint=='posture')&(contrasts.dataset!='pooled')&(contrasts.block_length==5)]
    pairs=[(1,3),(2,3),(3,4),(3,5),(3,6),(1,7),(2,8)]
    fig,axes=plt.subplots(1,2,figsize=(10.3,4.6),sharex=True,sharey=True)
    for ax,model in zip(axes,COLORS):
        for dataset,offset,color in [('coverall',-.12,'#245B7B'),('everyday',.12,'#BD6630')]:
            ss=[d[(d.model==model)&(d.dataset==dataset)&(d.condition_a==a)&(d.condition_b==b)].iloc[0] for a,b in pairs]
            values=np.array([r.estimate_pp for r in ss]);low=np.array([r.low for r in ss]);high=np.array([r.high for r in ss]);y=np.arange(7)+offset
            ax.errorbar(values,y,xerr=np.vstack([values-low,high-values]),fmt='o',color=color,capsize=2,markersize=4,label=SETS[dataset])
        ax.axvline(0,color='#777',lw=.8);ax.set_title(model);ax.set_xlabel('Change in exact code agreement (pp)');ax.set_yticks(range(7),[f'{a} to {b}' for a,b in pairs]);ax.grid(axis='x',color='#eee');ax.set_xlim(-20,32)
    axes[0].invert_yaxis()
    axes[0].set_ylabel('Condition comparison');handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=2,loc='lower center',frameon=False,bbox_to_anchor=(.5,-.015));fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(OUT/'figure4_information_changes.png',bbox_inches='tight');plt.close(fig)

    lux=read('illuminance_performance');lux=lux[lux.reference=='both']
    fig,axes=plt.subplots(2,2,figsize=(10.2,7.5),sharex=True,sharey=True)
    for i,dataset in enumerate(SETS):
        for j,model in enumerate(COLORS):
            ax=axes[i,j];arr=np.array([[lux[(lux.dataset==dataset)&(lux.model==model)&(lux.combination==c)&(lux.illuminance_lux==v)].iloc[0].exact_pct for v in [100,250,400]] for c in range(1,9)],float)
            ax.imshow(arr,vmin=0,vmax=60,cmap='Blues',aspect='auto')
            for rr in range(8):
                for cc in range(3):ax.text(cc,rr,f'{arr[rr,cc]:.1f}',ha='center',va='center',color='white' if arr[rr,cc]>35 else 'black',fontsize=10)
            ax.set_title(f'{SETS[dataset]}  {model}');ax.set_xticks([0,1,2],['100 lux','250 lux','400 lux']);ax.tick_params(labelbottom=True);ax.set_yticks(range(8),[f'C{c}' for c in range(1,9)])
    fig.subplots_adjust(left=.06,right=.89,bottom=.07,top=.94,hspace=.22,wspace=.2);cax=fig.add_axes([.925,.13,.018,.72]);fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,60),cmap='Blues'),cax=cax,label='Exact codes / illuminance group (%)');fig.savefig(OUT/'figureS1_lux_profiles.png',bbox_inches='tight');plt.close(fig)
    print('Four numerical figure files generated')

if __name__=='__main__':main()
