import math
import numpy as np
from scipy.special import gammaln
from sklearn.metrics import cohen_kappa_score

# Cited source table: see ../owas-data-results/reference/REFERENCE_PROVENANCE.md.
# Only load 1 is evaluated in this study.
RISK_ROWS = [
 [1,1,1, 1,1,1, 1,1,1, 2,2,2, 2,2,2, 1,1,1, 1,1,1],
 [1,1,1, 1,1,1, 1,1,1, 2,2,2, 2,2,2, 1,1,1, 1,1,1],
 [1,1,1, 1,1,1, 1,1,1, 2,2,3, 2,2,3, 1,1,1, 1,1,2],
 [2,2,3, 2,2,3, 2,2,3, 3,3,3, 3,3,3, 2,2,2, 2,3,3],
 [2,2,3, 2,2,3, 2,3,3, 3,4,4, 3,4,4, 3,3,4, 2,3,4],
 [3,3,4, 2,2,3, 3,3,3, 3,4,4, 4,4,4, 4,4,4, 2,3,4],
 [1,1,1, 1,1,1, 1,1,2, 3,3,3, 4,4,4, 1,1,1, 1,1,1],
 [2,2,3, 1,1,1, 1,1,2, 4,4,4, 4,4,4, 3,3,3, 1,1,1],
 [2,2,3, 1,1,1, 2,3,3, 4,4,4, 4,4,4, 4,4,4, 1,1,1],
 [2,3,3, 2,2,3, 2,2,3, 4,4,4, 4,4,4, 4,4,4, 2,3,4],
 [3,3,4, 2,3,4, 3,3,4, 4,4,4, 4,4,4, 4,4,4, 2,3,4],
 [4,4,4, 2,3,4, 3,3,4, 4,4,4, 4,4,4, 4,4,4, 2,3,4],
]

LOOKUP = {f'{b}{a}{l}1':RISK_ROWS[(b-1)*3+a-1][(l-1)*3]
          for b in range(1,5) for a in range(1,4) for l in range(1,8)}

def wilson(k,n):
    if n==0: return (None,None)
    z=1.959963984540054; p=k/n; den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return 100*(mid-half),100*(mid+half)

def kappa(x,y,weighted=False):
    if len(x)==0: return None
    labs=list(range(1,5)) if weighted else sorted(set(x)|set(y))
    count=np.zeros((len(labs),len(labs))); pos={v:i for i,v in enumerate(labs)}
    for a,b in zip(x,y): count[pos[a],pos[b]]+=1
    n=count.sum(); obs=count/n; exp=np.outer(count.sum(1),count.sum(0))/n**2
    if weighted:
        w=np.abs(np.arange(4)[:,None]-np.arange(4)[None,:])/3
        ans=None if np.sum(w*exp)==0 else 1-np.sum(w*obs)/np.sum(w*exp)
    else:
        pe=np.trace(exp);ans=None if abs(1-pe)<1e-14 else (np.trace(obs)-pe)/(1-pe)
    sk=cohen_kappa_score(x,y,labels=labs,weights='linear' if weighted else None) if len(labs)>1 else float('nan')
    if ans is not None: assert abs(ans-sk)<1e-10,(ans,sk)
    return ans

def holm(ps):
    ps=np.asarray(ps);order=np.argsort(ps);m=len(ps);out=np.ones(m);prev=0
    for rank,i in enumerate(order):
        prev=max(prev,(m-rank)*ps[i]);out[i]=min(1,prev)
    return out.tolist()

def ffh(success,totals):
    """Two-sided probability-ordered exact 3x2 conditional test, all tables."""
    n1,n2,n3=map(int,totals);k=sum(success);N=sum(totals)
    def lc(n,x): return gammaln(n+1)-gammaln(x+1)-gammaln(n-x+1)
    logden=lc(N,k)
    obs=sum(lc(n,x) for n,x in zip(totals,success))-logden
    probs=[]
    for a in range(max(0,k-n2-n3),min(n1,k)+1):
        for b in range(max(0,k-a-n3),min(n2,k-a)+1):
            c=k-a-b;lp=lc(n1,a)+lc(n2,b)+lc(n3,c)-logden
            if lp<=obs+1e-10: probs.append(math.exp(lp))
    return min(1.,math.fsum(probs))
