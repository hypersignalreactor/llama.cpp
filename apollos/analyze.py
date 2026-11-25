import csv, math, sys, numpy as np
def load(n):
    R=[r for r in csv.DictReader(open('out/%s.csv'%n)) if int(r['pos'])>=4]
    g=lambda c: np.array([float(r[c]) for r in R])
    D={c+str(k): g(c+str(k)) for c in ['top','ptop','ent','ptrue','gate'] for k in range(1,5)}; D['next']=g('next'); return D
def sim(D, exit_k):
    n=len(exit_k); idx=np.arange(n)
    top=np.stack([D['top%d'%k] for k in range(1,5)],1); pt=np.stack([D['ptrue%d'%k] for k in range(1,5)],1)
    t=top[idx,exit_k-1]; p=pt[idx,exit_k-1]
    return exit_k.mean(), (t==D['top4']).mean()*100, (t==D['next']).mean()*100, -np.log(np.maximum(p,1e-12)).mean()
def gate_exit(D,q):
    lam=[1/(1+np.exp(-D['gate%d'%k])) for k in (1,2,3)]; rem=np.ones_like(lam[0]); cdf=np.zeros_like(rem); ex=np.full(len(rem),4)
    for k in (1,2,3):
        cdf=cdf+lam[k-1]*rem; rem=rem*(1-lam[k-1]); ex=np.where((ex==4)&(cdf>=q),k,ex)
    return ex
def conf_exit(D,tau,kmin=1):
    ex=np.full(len(D['next']),4)
    for k in (3,2,1):
        if k>=kmin: ex=np.where(D['ptop%d'%k]>=tau,k,ex)
    return ex
def stable_exit(D,tau=0.0):
    ex=np.full(len(D['next']),4)
    for k in (3,2): ex=np.where((D['top%d'%k]==D['top%d'%(k-1)])&(D['ptop%d'%k]>=tau),k,ex)
    return ex
names=['kjv','qa','reason','prose']; ALL={n:load(n) for n in names}
ALL['ALL']={k:np.concatenate([ALL[n][k] for n in names]) for k in ALL['kjv']}
for n,D in ALL.items():
    print('\n== %s (%d tokens)'%(n,len(D['next'])))
    print(' fixed depth: loops, agree-with-4 %, top1-correct %, NLL')
    for k in range(1,5): print('   L%d  %.2f  %6.2f  %6.2f  %.4f'%((k,)+sim(D,np.full(len(D['next']),k))))
    lam=[1/(1+np.exp(-D['gate%d'%k])).mean() for k in (1,2,3)]; print(' mean gate lambda at loops 1-3: %.3f %.3f %.3f'%tuple(lam))
    if n!='ALL': continue
    for lab,f,grid in [('gate cdf>=q',gate_exit,[0.3,0.5,0.7,0.8,0.9,0.95,0.99]),('confidence p>=t',conf_exit,[0.5,0.7,0.8,0.9,0.95,0.99]),('conf, min 2 loops',lambda D,t:conf_exit(D,t,2),[0.5,0.7,0.9,0.95,0.99]),('stable+conf',stable_exit,[0.0,0.5,0.7,0.9,0.95])]:
        print(' policy %s: param, avg loops, agree %%, correct %%, NLL'%lab)
        for q in grid: print('   %-5s %.2f  %6.2f  %6.2f  %.4f'%((q,)+sim(D,f(D,q))))
# per-dataset at chosen points
print('\nper-dataset: policy -> avg loops / agree%')
for lab,f in [('gate q=0.9',lambda D:gate_exit(D,0.9)),('conf 0.9',lambda D:conf_exit(D,0.9)),('stable+conf .7',lambda D:stable_exit(D,0.7))]:
    print(' '+lab+': '+' | '.join('%s %.2f/%.1f'%((n,)+sim(ALL[n],f(ALL[n]))[:2]) for n in names))
