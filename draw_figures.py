"""Draw manuscript Figures 2–4 from the replay outputs (run reproduce.py first)."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
base=Path(__file__).resolve().parent
out=pd.read_csv(base/'reproduced/paired_test_transitions.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
dest=base/'reproduced'/'figures'
dest.mkdir(exist_ok=True, parents=True)
def save(fig,name):
    for fmt in ('svg','png'):
        fig.savefig(dest/(name+'.'+fmt),dpi=300,bbox_inches='tight',pad_inches=.09)
    plt.close(fig)
fig,ax=plt.subplots(figsize=(7.15,3.18))
colors=['#20646b','#b6d9dc','#92482f']
for y,row in enumerate(out.itertuples()):
    for start,n,color in [(0,-row.lost,colors[2]),(0,row.corrected,colors[0]),(row.corrected,row.completed,colors[1])]:
        if not n:continue
        ax.barh(y,n,left=start,height=.58,color=color,edgecolor='white',linewidth=.5)
        ax.text(start+n/2,y,str(abs(n)),ha='center',va='center',fontsize=10,color='#13252c' if color==colors[1] else 'white')
ax.axvline(0,color='#273b42',linewidth=.9)
ax.set_yticks(range(4),[r.archive+'  '+r.label for r in out.itertuples()],fontsize=9);ax.invert_yaxis()
ax.set_xlim(-8,17);ax.set_xticks([-5,0,5,10,15],['5','0','5','10','15']);ax.set_xlabel('Number of observations')
ax.tick_params(axis='y',length=0);ax.spines[['right','top','left']].set_visible(False)
ax.legend([Line2D([],[],lw=7,color=c) for c in [colors[2],colors[0],colors[1]]],['Match lost','Wrong code corrected','Missing code completed correctly'],frameon=False,loc='upper right',bbox_to_anchor=(1,-.28),ncol=3,fontsize=8,handlelength=1)
save(fig,'Figure3_Paired_Transitions')
fig,ax=plt.subplots(figsize=(7.45,4.25));ax.set_xlim(0,10);ax.set_ylim(0,6.1);ax.axis('off')
def box(x,y,w,h,label,fill):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',facecolor=fill,edgecolor='#607078',lw=.9))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9,linespacing=1.25)
def line(a,b):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',color='#354a52',lw=1.1,shrinkA=2,shrinkB=2))
box(.12,4.8,9.76,.85,'Front and side photographs\n181 development observations  •  113 observations from a separate test recording','#f1f4f5')
box(.12,2.95,4.52,1.03,'Image branch\nFrozen features → component classifiers\nTrained on development expert codes','#e1f0ee')
box(5.27,2.95,4.61,1.03,'Multimodal AI branch\nArchived codes for the back, arms and legs\nfrom one model under each input condition','#edf0f5')
line((2.37,4.75),(2.37,4.04));line((7.65,4.75),(7.65,4.04))
box(.12,.92,4.52,1.1,'Hybrid coding\nCombine AI and image predictions using\nexpert labels, or fill missing codes by rule','#e1f0ee')
box(5.27,.92,4.61,1.1,'AI codes alone\nTrain a classifier on development expert\ncodes, or combine by majority vote','#edf0f5')
line((2.37,2.9),(2.37,2.08));line((7.5,2.9),(7.5,2.08));line((6.15,2.9),(4.05,2.08))
ax.text(5,.32,'Scoring on the test recording used expert consensus only after prediction.',ha='center',fontsize=9)
save(fig,'Figure2_Study_Workflow')

outcomes=pd.read_csv(base/'reproduced/test_outcomes.csv')
selection = [
    ('M56','raw_condition_3','Paired-image AI'),
    ('M56','ai_ML_stacking','3 AI plus image logistic'),
    ('M61','raw_condition_3','Paired-image AI'),
    ('M61','AI3_Logistic','3 AI logistic'),
    ('M61','AI8_RBF_SVM','8 AI RBF-SVM'),
    ('M61','AI8ML_RBF_SVM','8 AI plus image RBF-SVM'),
    ('M61','ai_NA_filled_by_ML','Fill AI gaps with image ML'),
]
plot_rows=outcomes.set_index(['model','method']).loc[[(m,t) for m,t,label in selection]]
fig,ax=plt.subplots(figsize=(7.1,4.1))
left=pd.Series([0]*len(selection))
for field,label,colour in [('retained','Predicted AC3/4','#347D69'),('high_low','Predicted AC1/2','#B54C4C'),('high_NA','Unresolved','#B9BFC5')]:
    counts=plot_rows[field].reset_index(drop=True)
    ax.barh(range(len(selection)),counts,left=left,color=colour,label=label,height=.63)
    for y,count in enumerate(counts):
        if count:
            ax.text(left[y]+count/2,y,str(count),ha='center',va='center',fontsize=9,color='white' if field!='high_NA' else '#202020')
    left=left+counts
ax.set_yticks(range(len(selection)),[f'{model}  {label}' for model,_,label in selection],fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0,40)
ax.set_xlabel('High-priority observations in the held-out recording (n = 40)')
ax.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.45,-.16),fontsize=9)
ax.set_title('Predicted intervention priority for expert categories 3 and 4',fontsize=11,fontweight='bold',pad=10)
fig.tight_layout()
save(fig,'Figure4_High_Category_Outcomes')
