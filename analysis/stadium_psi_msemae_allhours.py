"""Stadium psi at hour 23:00, single 75/25 link split, NO folds.
Two objectives, both AVERAGED over box links (not summed):
  - MAE  : mean_link |F_i - v_i|        (reported x1e6)
  - MSE  : mean_link (F_i - v_i)^2       (reported x1e12)
Optimize psi separately for each objective; SP and TP.
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
ORIGINS_NPZ = str(INPUTS / "origins_results.npz")
SMOOTH_NPZ  = str(INPUTS / "popularity_results_smoothed_osm_gamma020.npz")
GRAPH_JSON  = str(INPUTS / "city_graph_full.json")
NETWORK_XML = str(INPUTS / "slc_network.xml")
OUT_JSON = RESULTS / "stadium_psi_msemae_allhours.json"
EPS=1e-6; TOL=1e-5; MAXI=200
SP_ALPHA=0.0508; SP_AS=1.7513; SP_AL=0.4358
BETA=0.102; RHO=0.101
TP_ASU=4.6489; TP_ASD=-0.5722; TP_ALU=1.1607; TP_ALD=-0.1673
LAPLACE=0.01; GAMMA_DIFF=0.20
BOX=(40.7468,40.7748,-111.8666,-111.8306)
GAMMAS=np.round(np.linspace(-0.012,0.04,53),4)
SEED=0
HOURS=list(range(18,24))
def get_speed_lane(g, links):
    s,l=[],[]
    for lid in links:
        e=g["links"].get(lid,{}); s.append(e.get("speed",11.17)); l.append(e.get("lanes",1.0))
    s=np.array(s,float); l=np.array(l,float)
    if s.mean()>0: s/=s.mean()
    if l.mean()>0: l/=l.mean()
    return s,l
def build_per_source(g,links,idx):
    adj=g.get("adjacency",{}); succ=[None]*len(links)
    for i,lid in enumerate(links):
        sl=[idx[o] for o in adj.get(lid,[]) if o in idx]
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
W={}
def _init():
    os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
    g=json.load(open(GRAPH_JSON)); links=list(g["links"].keys()); N=len(links)
    idx={l:i for i,l in enumerate(links)}
    speeds,lanes=get_speed_lane(g,links); Pdiff=build_P_diff(g,links,idx)
    nodes={}
    for _,el in ET.iterparse(NETWORK_XML,events=("start",)):
        if el.tag=="node": nodes[el.get("id")]=(float(el.get("x")),float(el.get("y")))
        else: el.clear()
    box=set()
    for _,el in ET.iterparse(NETWORK_XML,events=("start",)):
        if el.tag!="link": el.clear(); continue
        f,t=el.get("from"),el.get("to")
        if f in nodes and t in nodes:
            x1,y1=nodes[f]; x2,y2=nodes[t]; mlat=0.5*(y1+y2); mlon=0.5*(x1+x2)
            if BOX[0]<=mlat<=BOX[1] and BOX[2]<=mlon<=BOX[3]: box.add(el.get("id"))
        el.clear()
    box_idx=np.array(sorted(idx[c] for c in box if c in idx),np.int64)
    in_box=np.zeros(N,bool); in_box[box_idx]=True
    succ=build_per_source(g,links,idx); dm=np.array([s.size==0 for s in succ],bool)
    P_sp=build_P(N,succ,np.power(speeds,SP_AS)*np.power(lanes,SP_AL))
    P_up=build_P(N,succ,np.power(speeds,TP_ASU)*np.power(lanes,TP_ALU))
    P_dn=build_P(N,succ,np.power(speeds,TP_ASD)*np.power(lanes,TP_ALD))
    bO=load_popularity_npz(ORIGINS_NPZ); bP=load_popularity_npz(SMOOTH_NPZ)
    Oproj=np.array([idx.get(str(x),-1) for x in bO["link_ids"]],np.int64); Oval=Oproj>=0
    Otidx={str(t):i for i,t in enumerate(bO["times"])}
    Pproj=np.array([idx.get(str(x),-1) for x in bP["link_ids"]],np.int64); Pval=Pproj>=0
    Ptidx={str(t):i for i,t in enumerate(bP["times"])}
    W.update(dict(N=N,box_idx=box_idx,in_box=in_box,dm=dm,Pdiff=Pdiff,
                  P_sp=P_sp,P_up=P_up,P_dn=P_dn,
                  bO=bO,Oproj=Oproj,Oval=Oval,Otidx=Otidx,
                  bP=bP,Pproj=Pproj,Pval=Pval,Ptidx=Ptidx))
def worker(payload):
    hour,k,slot08,slot15=payload
    N=W["N"]; box_idx=W["box_idx"]; in_box=W["in_box"]; dm=W["dm"]; nb=len(box_idx)
    ti=W["Otidx"].get(slot08)
    raw=W["bO"]["matrix"].getrow(ti).toarray().ravel().astype(float) if ti is not None else None
    E=diffuse(raw,W["Oproj"],W["Oval"],N,W["Pdiff"])
    row=W["bP"]["matrix"].getrow(W["Ptidx"][slot15]).toarray().ravel().astype(float)
    F=np.zeros(N); np.add.at(F,W["Pproj"][W["Pval"]],row[W["Pval"]]); s=F.sum()
    if s>0: F/=s
    Fb=F[box_idx]; b=float(E[box_idx].sum()); a=1.0-b
    ng=len(GAMMAS)
    out={m:np.zeros((ng,nb),np.float32) for m in ("sp_abs","sp_sq","tp_abs","tp_sq")}
    for gi,gam in enumerate(GAMMAS):
        q=E.copy(); q[~in_box]=q[~in_box]*(1.0-gam/a); q[box_idx]=q[box_idx]*(1.0+gam/b)
        vs=sp_iter(W["P_sp"],q,dm)[box_idx]; vt=tp_iter(W["P_up"],W["P_dn"],q,dm)[box_idx]
        out["sp_abs"][gi]=(np.abs(Fb-vs)*1e6).astype(np.float32)
        out["tp_abs"][gi]=(np.abs(Fb-vt)*1e6).astype(np.float32)
        out["sp_sq"][gi]=(((Fb-vs)**2)*1e12).astype(np.float32)
        out["tp_sq"][gi]=(((Fb-vt)**2)*1e12).astype(np.float32)
    return hour,k,out
def main():
    t0=time.time()
    payloads=[]
    for hour in HOURS:
        for k,mn in enumerate(range(0,60,5)):
            payloads.append((hour,k,f"2018-09-08 {hour:02d}:{mn:02d}:00",f"2018-09-15 {hour:02d}:{mn:02d}:00"))
    mw=int(os.environ.get("TPPR_PARALLEL", min(8,(os.cpu_count() or 4))))
    print(f"[start] hours {HOURS[0]}-{HOURS[-1]}  {len(payloads)} bin-tasks x {len(GAMMAS)} psi, {mw} workers",flush=True)
    res={h:{} for h in HOURS}
    with ProcessPoolExecutor(max_workers=mw,initializer=_init) as ex:
        futs=[ex.submit(worker,p) for p in payloads]; done=0
        for f in as_completed(futs):
            hour,k,o=f.result(); res[hour][k]=o; done+=1
            if done%12==0: print(f"  {done}/{len(payloads)} bin-tasks",flush=True)
    g0=int(np.argmin(np.abs(GAMMAS)))
    out={"hours":HOURS,"gammas":GAMMAS.tolist(),"models":{}}
    def fit(M,tr,te):
        tm=M[:,:,tr].mean(axis=(0,2)); gi=int(np.argmin(tm)); gam=float(GAMMAS[gi])
        tr_pb=M[:,gi,tr].mean(axis=1); te_pb=M[:,gi,te].mean(axis=1)
        tr0_pb=M[:,g0,tr].mean(axis=1); te0_pb=M[:,g0,te].mean(axis=1)
        return dict(psi=gam,train=float(tr_pb.mean()),test=float(te_pb.mean()),
                    train0=float(tr0_pb.mean()),test0=float(te0_pb.mean()))
    print("\n  MAE = mean_link |F-v| x1e6     MSE = mean_link (F-v)^2 x1e12   (single 75/25 split)\n")
    print("  hour | model | MAE psi*   MAE test  (psi=0)  | MSE psi*   MSE test    (psi=0)")
    print("  -----+-------+--------------------------------+-------------------------------")
    for hour in HOURS:
        nb=len(res[hour])
        stacks={m:np.stack([res[hour][k][m] for k in range(nb)]) for m in ("sp_abs","sp_sq","tp_abs","tp_sq")}
        nlink=stacks["sp_abs"].shape[2]
        rng=np.random.default_rng(SEED); perm=rng.permutation(nlink)
        ntr=int(round(0.75*nlink)); tr=perm[:ntr]; te=perm[ntr:]
        out["models"][hour]={}
        for model in ("sp","tp"):
            ra=fit(stacks[f"{model}_abs"],tr,te); rs=fit(stacks[f"{model}_sq"],tr,te)
            out["models"][hour][model]={"mae":ra,"mse":rs}
            print("  %4d | %-5s | %+0.4f  %8.3f  %8.3f | %+0.4f  %9.2f  %9.2f"
                  %(hour if model=="sp" else 0, model.upper(),
                    ra["psi"],ra["test"],ra["test0"], rs["psi"],rs["test"],rs["test0"]))
        print("  -----+-------+--------------------------------+-------------------------------")
    # ---- FULL WINDOW: pool all 72 bins (18:00-23:55) into one fit ----
    allbins={}
    for m in ("sp_abs","sp_sq","tp_abs","tp_sq"):
        allbins[m]=np.concatenate([np.stack([res[h][k][m] for k in range(len(res[h]))]) for h in HOURS],axis=0)
    nlink=allbins["sp_abs"].shape[2]
    rng=np.random.default_rng(SEED); perm=rng.permutation(nlink)
    ntr=int(round(0.75*nlink)); tr=perm[:ntr]; te=perm[ntr:]
    def fit_full(M):
        tm=M[:,:,tr].mean(axis=(0,2)); gi=int(np.argmin(tm)); gam=float(GAMMAS[gi])
        tr_pb=M[:,gi,tr].mean(axis=1); te_pb=M[:,gi,te].mean(axis=1); base_pb=M[:,g0,:].mean(axis=1)
        return dict(psi=gam,train=float(tr_pb.mean()),test=float(te_pb.mean()),base=float(base_pb.mean()))
    out["full_window"]={}
    print("\n=== FULL WINDOW 18:00-23:55 pooled (%d bins), MSE x1e12 ==="%(allbins["sp_sq"].shape[0]))
    print("| model | psi* | train | test | psi=0 |")
    for model in ("sp","tp"):
        r=fit_full(allbins[f"{model}_sq"]); out["full_window"][model]=r
        print("| %s | %+0.4f | %.2f | %.2f | %.2f |"%(model.upper(),r["psi"],r["train"],r["test"],r["base"]))
    OUT_JSON.parent.mkdir(parents=True,exist_ok=True); OUT_JSON.write_text(json.dumps(out,indent=2))
    print("\n[done] %ds saved %s"%(time.time()-t0,OUT_JSON))

if __name__=="__main__":
    main()
