"""9th & 9th festival: learn the prior mass-transfer gamma, optionally on
top of surgery. One run per MODE (env TPPR_MODE):
  gamma_only  : full network, no closure handling; box = full festival box.
  gamma_surg1 : remove the 14 closed links; prior & popularity zeroed+renorm
                on the open links; box = festival box minus closed.
  gamma_surg2 : same removal/prior; popularity redistributed onto the box;
                box = festival box minus closed.
In every mode we additionally reweight the (mode-adjusted) prior by gamma,
moving prior mass into the eval box, and learn gamma to minimise the box
MRE. Train/test over the 96 time bins (2/3-1/3) and over the box links
(75/25), each with 5-fold CV. Previous-week prior (Sat 2018-09-08); target
= festival-day popularity (Sat 2018-09-15). SP and TP separately.
gamma=0 row = the plain mode (none / surgery1 / surgery2).
Output: results/festival_gamma_<MODE>.json
"""
import json, os, sys, time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.io import load_popularity_npz

INPUTS  = Path(os.environ.get("TPPR_INPUTS",  str(ROOT / "data")))
RESULTS = Path(os.environ.get("TPPR_RESULTS", str(ROOT / "results/origin_pop_ladder")))
ORIGINS_NPZ=str(INPUTS/"origins_results.npz"); SMOOTH_NPZ=str(INPUTS/"popularity_results_smoothed_osm_gamma020.npz")
GRAPH_JSON=str(INPUTS/"city_graph_full.json"); NETWORK_XML=str(INPUTS/"slc_network.xml")
MODE=os.environ.get("TPPR_MODE","gamma_only")
OUT_JSON=RESULTS/f"festival_gamma_{MODE}.json"

EPS=1e-6; TOL=1e-5; MAXI=200
SP_ALPHA=0.0508; SP_AS=1.7513; SP_AL=0.4358
BETA=0.102; RHO=0.101
TP_ASU=4.6489; TP_ASD=-0.5722; TP_ALU=1.1607; TP_ALD=-0.1673
LAPLACE=0.01; GAMMA_DIFF=0.20
B1=(40.749590,40.749940,-111.868348,-111.863884)   # EW 900 S
B2=(40.747908,40.752069,-111.865526,-111.865140)   # NS 900 E
BOX=(40.741419,40.756455,-111.876984,-111.853615)   # festival eval box
NG=29; SEED=0; N_TRAIN_BINS=64; TRAIN_FRAC_LINKS=0.75; KFOLD=5


def get_speed_lane(g,links):
    s,l=[],[]
    for lid in links:
        e=g["links"].get(lid,{}); s.append(e.get("speed",11.17)); l.append(e.get("lanes",1.0))
    s=np.array(s,float); l=np.array(l,float)
    if s.mean()>0: s/=s.mean()
    if l.mean()>0: l/=l.mean()
    return s,l

def build_per_source(g,links,idx,closed):
    adj=g.get("adjacency",{}); succ=[None]*len(links)
    for i,lid in enumerate(links):
        if lid in closed: succ[i]=np.empty(0,np.int64); continue
        sl=[idx[o] for o in adj.get(lid,[]) if o in idx and o not in closed]
        succ[i]=np.array(sl,np.int64) if sl else np.empty(0,np.int64)
    return succ

def build_P(N,succ,dw):
    r,c,d=[],[],[]
    for i in range(N):
        ix=succ[i]
        if ix.size==0: continue
        w=dw[ix]; tot=w.sum()
        if tot<=0: continue
        w=w/tot
        for k,j in enumerate(ix): r.append(int(j)); c.append(i); d.append(float(w[k]))
    return csr_matrix((d,(r,c)),shape=(N,N))

def build_P_diff(g,links,idx):
    adj=g.get("adjacency",{}); r,c,d=[],[],[]
    for lid,outs in adj.items():
        if lid not in idx: continue
        i=idx[lid]; v=[idx[o] for o in outs if o in idx]
        if not v: continue
        w=1.0/len(v)
        for j in v: r.append(i); c.append(j); d.append(w)
    return csr_matrix((d,(r,c)),shape=(len(links),len(links)))

def sp_iter(P,prior,dm):
    pf=prior.astype(float,copy=True); v=pf.copy(); oa=1-SP_ALPHA
    for _ in range(MAXI):
        Pv=P@v+float(v[dm].sum())*pf; vn=SP_ALPHA*pf+oa*Pv; s=vn.sum()
        if s>0: vn/=s
        if float(np.abs(vn-v).sum())<TOL: return vn
        v=vn
    return v

def tp_iter(Pu,Pd,E,dm):
    E=E.astype(float,copy=True); N=E.shape[0]; vu=E.copy(); vd=np.zeros(N)
    for _ in range(MAXI):
        sd=float(vd.sum())
        Pvu=Pu@vu+float(vu[dm].sum())*E; Pvd=Pd@vd+float(vd[dm].sum())*E
        vun=(1-BETA)*Pvu+RHO*E*sd; vdn=BETA*vu+(1-RHO)*Pvd
        s=float(vun.sum()+vdn.sum())
        if s>0: vun/=s; vdn/=s
        diff=float(np.abs(vun-vu).sum()+np.abs(vdn-vd).sum()); vu,vd=vun,vdn
        if diff<TOL: break
    return vu+vd

def diffuse(raw,proj,valid,N,Pd):
    c=np.full(N,LAPLACE)
    if raw is not None:
        add=np.zeros(N); np.add.at(add,proj[valid],raw[valid]); c=c+add
    cd=(1-GAMMA_DIFF)*c+GAMMA_DIFF*(Pd@c); s=cd.sum()
    if s>0: cd/=s
    return cd


def build():
    g=json.load(open(GRAPH_JSON)); links=list(g["links"].keys()); N=len(links)
    idx={l:i for i,l in enumerate(links)}
    speeds,lanes=get_speed_lane(g,links); Pdiff=build_P_diff(g,links,idx)
    nodes={}
    for _,el in ET.iterparse(NETWORK_XML,events=("start",)):
        if el.tag=="node": nodes[el.get("id")]=(float(el.get("x")),float(el.get("y")))
        else: el.clear()
    closed=set(); box=set()
    for _,el in ET.iterparse(NETWORK_XML,events=("start",)):
        if el.tag!="link": el.clear(); continue
        f,t=el.get("from"),el.get("to")
        if f in nodes and t in nodes:
            x1,y1=nodes[f]; x2,y2=nodes[t]; dx,dy=x2-x1,y2-y1
            mlat=0.5*(y1+y2); mlon=0.5*(x1+x2); lid=el.get("id")
            b1=(B1[0]<=mlat<=B1[1] and B1[2]<=mlon<=B1[3] and abs(dx)>1.5*abs(dy))
            b2=(B2[0]<=mlat<=B2[1] and B2[2]<=mlon<=B2[3] and abs(dy)>1.5*abs(dx))
            if b1 or b2: closed.add(lid)
            if BOX[0]<=mlat<=BOX[1] and BOX[2]<=mlon<=BOX[3]: box.add(lid)
        el.clear()
    if MODE=="gamma_only": closed=set()
    succ=build_per_source(g,links,idx,closed); dm=np.array([s.size==0 for s in succ],bool)
    P_sp=build_P(N,succ,np.power(speeds,SP_AS)*np.power(lanes,SP_AL))
    P_up=build_P(N,succ,np.power(speeds,TP_ASU)*np.power(lanes,TP_ALU))
    P_dn=build_P(N,succ,np.power(speeds,TP_ASD)*np.power(lanes,TP_ALD))
    closed_idx=np.array(sorted(idx[c] for c in closed if c in idx),np.int64)
    box_idx=np.array(sorted(idx[c] for c in box if c in idx and c not in closed),np.int64)
    in_box=np.zeros(N,bool); in_box[box_idx]=True
    is_closed=np.zeros(N,bool); is_closed[closed_idx]=True
    A_idx=np.where(~in_box & ~is_closed)[0]
    bO=load_popularity_npz(ORIGINS_NPZ); bP=load_popularity_npz(SMOOTH_NPZ)
    Oproj=np.array([idx.get(str(x),-1) for x in bO["link_ids"]],np.int64); Oval=Oproj>=0
    Otidx={str(t):i for i,t in enumerate(bO["times"])}
    Pproj=np.array([idx.get(str(x),-1) for x in bP["link_ids"]],np.int64); Pval=Pproj>=0
    Ptidx={str(t):i for i,t in enumerate(bP["times"])}
    return dict(N=N,box_idx=box_idx,A_idx=A_idx,closed_idx=closed_idx,dm=dm,Pdiff=Pdiff,
                P_sp=P_sp,P_up=P_up,P_dn=P_dn,bO=bO,Oproj=Oproj,Oval=Oval,Otidx=Otidx,
                bP=bP,Pproj=Pproj,Pval=Pval,Ptidx=Ptidx)

def zoc(v,closed_idx):
    x=v.copy(); x[closed_idx]=0.0; s=x.sum()
    if s>0: x/=s
    return x
def redist_to_box(v,closed_idx,box_idx):
    x=v.copy(); m=float(x[closed_idx].sum()); x[closed_idx]=0.0
    base=x[box_idx]; bs=float(base.sum())
    if bs>0: x[box_idx]=base+m*base/bs
    else: x[box_idx]=x[box_idx]+m/len(box_idx)
    return x

def prior0(Wd,slot):
    N=Wd["N"]; ti=Wd["Otidx"].get(slot)
    raw=Wd["bO"]["matrix"].getrow(ti).toarray().ravel().astype(float) if ti is not None else None
    E=diffuse(raw,Wd["Oproj"],Wd["Oval"],N,Wd["Pdiff"])
    if MODE!="gamma_only": E=zoc(E,Wd["closed_idx"])
    return E
def target0(Wd,slot):
    N=Wd["N"]; row=Wd["bP"]["matrix"].getrow(Wd["Ptidx"][slot]).toarray().ravel().astype(float)
    F=np.zeros(N); np.add.at(F,Wd["Pproj"][Wd["Pval"]],row[Wd["Pval"]]); s=F.sum()
    if s>0: F/=s
    if MODE=="gamma_surg1": F=zoc(F,Wd["closed_idx"])
    elif MODE=="gamma_surg2": F=redist_to_box(F,Wd["closed_idx"],Wd["box_idx"])
    return F


W={}
def _init():
    os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
    W.update(build())

def worker(payload):
    k,slot08,slot15,gammas=payload
    box_idx=W["box_idx"]; A_idx=W["A_idx"]; dm=W["dm"]; nb=len(box_idx)
    E0=prior0(W,slot08); F0=target0(W,slot15); Fb=F0[box_idx]
    b=float(E0[box_idx].sum()); a=1.0-b
    sp=np.zeros((len(gammas),nb),np.float32); tp=np.zeros((len(gammas),nb),np.float32)
    for gi,gam in enumerate(gammas):
        q=E0.copy(); q[A_idx]=q[A_idx]*(1.0-gam/a); q[box_idx]=q[box_idx]*(1.0+gam/b)
        vs=sp_iter(W["P_sp"],q,dm); vt=tp_iter(W["P_up"],W["P_dn"],q,dm)
        sp[gi]=(np.abs(Fb-vs[box_idx])/(Fb+EPS)).astype(np.float32)
        tp[gi]=(np.abs(Fb-vt[box_idx])/(Fb+EPS)).astype(np.float32)
    return k,sp,tp


def main():
    t0=time.time()
    slots15=[f"2018-09-15 {h:02d}:{m:02d}:00" for h in range(10,18) for m in range(0,60,5)]
    slots08=[f"2018-09-08 {h:02d}:{m:02d}:00" for h in range(10,18) for m in range(0,60,5)]
    nb=len(slots15)
    print(f"[start] MODE={MODE}  {nb} bins",flush=True)
    Wm=build()
    bb=np.array([float(prior0(Wm,s)[Wm["box_idx"]].sum()) for s in slots08])
    bmin=float(bb.min()); gmax=0.9*bmin
    gammas=np.linspace(-gmax,gmax,NG).tolist()
    print(f"  box links={len(Wm['box_idx'])}  closed={len(Wm['closed_idx'])}  b mean={bb.mean():.4f} min={bmin:.4f}  gmax={gmax:.4f}",flush=True)
    payloads=[(k,slots08[k],slots15[k],gammas) for k in range(nb)]
    mw=int(os.environ.get("TPPR_PARALLEL", min(8,(os.cpu_count() or 4))))
    res={}
    with ProcessPoolExecutor(max_workers=mw,initializer=_init) as ex:
        futs=[ex.submit(worker,p) for p in payloads]
        done=0
        for f in as_completed(futs):
            k,sp,tp=f.result(); res[k]=(sp,tp); done+=1
            if done%16==0: print(f"  {done}/{nb} bins",flush=True)
    G=np.array(gammas); g0=int(np.argmin(np.abs(G)))
    Esp=np.stack([res[k][0] for k in range(nb)]); Etp=np.stack([res[k][1] for k in range(nb)])
    nlink=Esp.shape[2]

    rng=np.random.default_rng(SEED)
    def fit_bins(M):  # M[bins,G,links] -> per-bin curve
        Mb=M.mean(2)  # [bins,G]
        perm=rng.permutation(nb); tr=perm[:N_TRAIN_BINS]; te=perm[N_TRAIN_BINS:]
        def f(train,test):
            tm=Mb[train].mean(0); gi=int(np.argmin(tm))
            return dict(gamma=float(G[gi]),train=float(tm[gi]),test=float(Mb[test,gi].mean()),
                        base_test=float(Mb[test,g0].mean()))
        single=f(tr,te)
        folds=np.array_split(perm,KFOLD); rows=[]
        for i in range(KFOLD):
            test=folds[i]; train=np.concatenate([folds[j] for j in range(KFOLD) if j!=i]); rows.append(f(train,test))
        cv=dict(mean_gamma=float(np.mean([r["gamma"] for r in rows])),
                mean_train=float(np.mean([r["train"] for r in rows])),
                mean_test=float(np.mean([r["test"] for r in rows])),
                mean_base=float(np.mean([r["base_test"] for r in rows])),folds=rows)
        return single,cv
    def fit_links(M):
        perm=rng.permutation(nlink); ntr=int(round(TRAIN_FRAC_LINKS*nlink)); tr=perm[:ntr]; te=perm[ntr:]
        def f(train,test):
            tm=M[:,:,train].mean(axis=(0,2)); gi=int(np.argmin(tm))
            return dict(gamma=float(G[gi]),train=float(tm[gi]),test=float(M[:,gi,test].mean()),
                        base_test=float(M[:,g0,test].mean()))
        single=f(tr,te)
        folds=np.array_split(perm,KFOLD); rows=[]
        for i in range(KFOLD):
            test=folds[i]; train=np.concatenate([folds[j] for j in range(KFOLD) if j!=i]); rows.append(f(train,test))
        cv=dict(mean_gamma=float(np.mean([r["gamma"] for r in rows])),
                mean_train=float(np.mean([r["train"] for r in rows])),
                mean_test=float(np.mean([r["test"] for r in rows])),
                mean_base=float(np.mean([r["base_test"] for r in rows])),folds=rows)
        return single,cv

    out={"mode":MODE,"n_bins":nb,"n_links":nlink,"gammas":gammas,
         "baseline_box_mre":{"SP":float(Esp[:,g0,:].mean()),"TP":float(Etp[:,g0,:].mean())},
         "bins":{},"links":{}}
    print(f"\n=== MODE={MODE} | baseline (gamma=0) box MRE: SP {out['baseline_box_mre']['SP']:.4f}  TP {out['baseline_box_mre']['TP']:.4f} ===")
    for name,M in [("SP",Esp),("TP",Etp)]:
        s,cv=fit_bins(M); out["bins"][name]={"single":s,"cv":cv}
        print("  BINS  %s | single gamma*=%+.4f train %.4f test %.4f (base %.4f) | CV gamma*=%+.4f test %.4f (base %.4f)"
              %(name,s["gamma"],s["train"],s["test"],s["base_test"],cv["mean_gamma"],cv["mean_test"],cv["mean_base"]))
    for name,M in [("SP",Esp),("TP",Etp)]:
        s,cv=fit_links(M); out["links"][name]={"single":s,"cv":cv}
        print("  LINKS %s | single gamma*=%+.4f train %.4f test %.4f (base %.4f) | CV gamma*=%+.4f test %.4f (base %.4f)"
              %(name,s["gamma"],s["train"],s["test"],s["base_test"],cv["mean_gamma"],cv["mean_test"],cv["mean_base"]))
    OUT_JSON.parent.mkdir(parents=True,exist_ok=True); OUT_JSON.write_text(json.dumps(out,indent=2))
    print(f"[done] {time.time()-t0:.0f}s saved {OUT_JSON}")


if __name__=="__main__":
    main()
