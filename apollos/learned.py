# A learned exit rule: logistic regression on cheap per-loop signals predicting "this loop's top-1 already equals the full-depth top-1".
import csv, numpy as np
exec(open('analyze.py').read().split("names=['kjv'")[0])
names=['kjv','qa','reason','prose']; Ds=[load(n) for n in names]
D={k:np.concatenate([d[k] for d in Ds]) for k in Ds[0]}; n=len(D['next']); rng=np.random.default_rng(7); perm=rng.permutation(n); tr,te=perm[:n//2],perm[n//2:]
def feats(k):
    f=[D['ptop%d'%k], D['ent%d'%k]/10, D['gate%d'%k]/5]
    if k>1: f+= [(D['top%d'%k]==D['top%d'%(k-1)]).astype(float), D['ptop%d'%k]-D['ptop%d'%(k-1)]]
    return np.stack(f+[np.ones(n)],1)
W={}
for k in (1,2,3):
    X=feats(k); y=(D['top%d'%k]==D['top4']).astype(float); w=np.zeros(X.shape[1])
    for it in range(4000):
        p=1/(1+np.exp(-X[tr]@w)); w-=0.5*(X[tr].T@(p-y[tr])/len(tr)+1e-4*w)
    W[k]=w; print('loop',k,'weights',np.round(w,2))
P={k:1/(1+np.exp(-feats(k)@W[k])) for k in (1,2,3)}
sub=lambda idx:{k:v[idx] for k,v in D.items()}
print('held-out half (%d tokens): rule, param, avg loops, agree %%, correct %%, NLL'%len(te))
for th in (0.9,0.95,0.98,0.99,0.995):
    ex=np.full(n,4)
    for k in (3,2,1): ex=np.where(P[k]>=th,k,ex)
    print('  learned %-6s %.2f %6.2f %6.2f %.4f'%((th,)+sim(sub(te),ex[te])))
for t in (0.7,0.8,0.9,0.95): print('  conf    %-6s %.2f %6.2f %6.2f %.4f'%((t,)+sim(sub(te),conf_exit(D,t)[te])))
print('  full depth    %.2f %6.2f %6.2f %.4f'%sim(sub(te),np.full(len(te),4)))
ex=conf_exit(D,0.9); print('exit histogram at conf 0.9 (loops 1-4):', [int((ex==k).sum()) for k in (1,2,3,4)])
