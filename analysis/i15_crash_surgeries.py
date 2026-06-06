"""I-15 NB Draper crash: no-surgery / surgery1 / surgery2, prev-week prior.

Closure = the non-expanded red set of i15_crash_event_links.pdf: the
single NB I-15 link 59076 (no neighbour expansion).

Window: 2018-09-20 07:00..11:55 (60 slots). Prior = previous-week origins
E_{b-7d} (same hours Thu 2018-09-13). Target = crash-day popularity.

Conditions:
  none     : full graph, plain (closed link only zeroed for scoring).
  surgery2 : remove the closed links from the network and rebuild P; move the
             closed links' PRIOR mass onto the eval box (proportional to the
             box prior); for the popularity target, drop the closed links and
             renormalize over the whole network.

Models: single-phase and two-phase up/down, speed+lane, no angle.
Scored (Overall MRE mean/dev over 60 slots) on the eval box and whole
network, open links only.

Output: results/origin_pop_ladder/i15_crash_surgeries.json
"""
import json, os, sys, time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.io import load_popularity_npz

ORIGINS_NPZ = str(ROOT / "data/origins_results.npz")
SMOOTH_NPZ  = str(ROOT / "data/popularity_results_smoothed_osm_gamma020.npz")
GRAPH_JSON  = str(ROOT / "data/city_graph_full.json")
NETWORK_XML = str(ROOT / "data/slc_network.xml")
OUT_JSON    = ROOT / "results" / "origin_pop_ladder" / "i15_crash_surgeries.json"

EPS=1e-6; TOL=1e-5; MAXI=200
SP_ALPHA=0.0508; SP_AS=1.7513; SP_AL=0.4358
BETA=0.102; RHO=0.101
TP_ASU=4.6489; TP_ASD=-0.5722; TP_ALU=1.1607; TP_ALD=-0.1673
LAPLACE=0.01; GAMMA=0.20
# crash seed closure box (NB I-15) + eval box (gen_i15_crash_event_links.py)
CLAT=(40.486,40.501); CLON=(-111.895,-111.890)
BOX=(40.476825,40.507853,-111.910002,-111.872202)


def get_speed_lane(g, links):
    s,l=[],[]
    for lid in links:
        e=g["links"].get(lid,{}); s.append(e.get("speed",11.17)); l.append(e.get("lanes",1.0))
    s=np.array(s,float); l=np.array(l,float)
    if s.mean()>0: s/=s.mean()
    if l.mean()>0: l/=l.mean()
    return s,l

def build_per_source(g,links,idx,closed=None):
    closed=closed or set(); adj=g.get("adjacency",{}); succ=[None]*len(links)
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

def sp_iter(P,prior):
    pf=prior.astype(float,copy=True); v=pf.copy(); oa=1-SP_ALPHA
    for _ in range(MAXI):
        vn=SP_ALPHA*pf+oa*(P@v); s=vn.sum()
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
    cd=(1-GAMMA)*c+GAMMA*(Pd@c); s=cd.sum()
    if s>0: cd/=s
    return cd


def main():
    t0=time.time()
    g=json.load(open(GRAPH_JSON)); links=list(g["links"].keys()); N=len(links)
    idx={l:i for i,l in enumerate(links)}
    speeds,lanes=get_speed_lane(g,links); Pdiff=build_P_diff(g,links,idx)
    adj=g.get("adjacency",{})
    nodes={}
    for _,el in ET.iterparse(NETWORK_XML,events=("start",)):
        if el.tag=="node": nodes[el.get("id")]=(float(el.get("x")),float(el.get("y")))
        else: el.clear()
    seed=set(); box=set()
    for _,el in ET.iterparse(NETWORK_XML,events=("start",)):
        if el.tag!="link": el.clear(); continue
        f,t=el.get("from"),el.get("to")
        if f in nodes and t in nodes:
            x1,y1=nodes[f]; x2,y2=nodes[t]; dx,dy=x2-x1,y2-y1
            mlat=0.5*(y1+y2); mlon=0.5*(x1+x2); fs=float(el.get("freespeed",0.0)); lid=el.get("id")
            if CLAT[0]<=mlat<=CLAT[1] and CLON[0]<=mlon<=CLON[1] and fs>=31.0 and abs(dy)>1.5*abs(dx) and dy>0:
                seed.add(lid)
            if BOX[0]<=mlat<=BOX[1] and BOX[2]<=mlon<=BOX[3]: box.add(lid)
        el.clear()
    closed=set(seed)   # non-expanded: seed NB I-15 link (59076) only, no neighbor expansion
    box-=closed
    closed_idx=np.array([idx[c] for c in closed if c in idx],np.int64)
    box_idx=np.array([idx[c] for c in box if c in idx],np.int64)
    cset=set(closed_idx.tolist())
    whole_idx=np.array([i for i in range(N) if i not in cset],np.int64)
    print(f"  N={N:,} closed={len(closed)} box(no-closed)={len(box_idx)}")
    print(f"  closed links: {sorted(closed)}")

    succ_o=build_per_source(g,links,idx,closed=None)
    succ_s=build_per_source(g,links,idx,closed=closed)
    dm_o=np.array([s.size==0 for s in succ_o],bool)
    dm_s=np.array([s.size==0 for s in succ_s],bool)
    w_sp=np.power(speeds,SP_AS)*np.power(lanes,SP_AL)
    w_up=np.power(speeds,TP_ASU)*np.power(lanes,TP_ALU)
    w_dn=np.power(speeds,TP_ASD)*np.power(lanes,TP_ALD)
    P_sp_o=build_P(N,succ_o,w_sp); P_sp_s=build_P(N,succ_s,w_sp)
    P_up_o=build_P(N,succ_o,w_up); P_dn_o=build_P(N,succ_o,w_dn)
    P_up_s=build_P(N,succ_s,w_up); P_dn_s=build_P(N,succ_s,w_dn)

    slots20=[f"2018-09-20 {h:02d}:{m:02d}:00" for h in range(7,12) for m in range(0,60,5)]
    slots13=[f"2018-09-13 {h:02d}:{m:02d}:00" for h in range(7,12) for m in range(0,60,5)]
    bO=load_popularity_npz(ORIGINS_NPZ); bP=load_popularity_npz(SMOOTH_NPZ)
    Oproj=np.array([idx.get(str(x),-1) for x in bO["link_ids"]],np.int64); Oval=Oproj>=0
    Otidx={str(t):i for i,t in enumerate(bO["times"])}
    Pproj=np.array([idx.get(str(x),-1) for x in bP["link_ids"]],np.int64); Pval=Pproj>=0
    Ptidx={str(t):i for i,t in enumerate(bP["times"])}
    def ovec(slot):
        ti=Otidx.get(slot)
        raw=bO["matrix"].getrow(ti).toarray().ravel().astype(float) if ti is not None else None
        return diffuse(raw,Oproj,Oval,N,Pdiff)
    def pvec(slot):
        row=bP["matrix"].getrow(Ptidx[slot]).toarray().ravel().astype(float)
        E=np.zeros(N); np.add.at(E,Pproj[Pval],row[Pval]); s=E.sum()
        if s>0: E/=s
        return E
    SAMEDAY = os.environ.get("TPPR_SAMEDAY")=="1"
    prior_slots = slots20 if SAMEDAY else slots13
    print(f"[load] {'same-day' if SAMEDAY else 'prev-week'} priors + crash-day targets (Sep 20) ...")
    E_prev=[ovec(s) for s in prior_slots]
    F=[pvec(s) for s in slots20]

    def zoc(v):
        x=v.copy(); x[closed_idx]=0.0; s=x.sum()
        if s>0: x/=s
        return x
    def redist_to_box(v):
        x=v.copy(); m=float(x[closed_idx].sum()); x[closed_idx]=0.0
        base=x[box_idx]; bs=float(base.sum())
        if bs>0: x[box_idx]=base+m*base/bs
        else:    x[box_idx]=x[box_idx]+m/len(box_idx)
        return x
    def mre(t,p,mask): return float(np.mean(np.abs(t[mask]-p[mask])/(t[mask]+EPS)))

    def run(model, cond):
        surg = cond in ("surgery2","surgery_box")
        pb,pw=[],[]
        for k in range(len(slots20)):
            E=E_prev[k]
            # surgery: move the closed links' PRIOR mass onto the eval box (proportional)
            prior = redist_to_box(E) if surg else E
            if model=="sp":
                v = sp_iter(P_sp_s if surg else P_sp_o, prior)
            else:
                v = tp_iter(P_up_s,P_dn_s,prior,dm_s) if surg else tp_iter(P_up_o,P_dn_o,prior,dm_o)
            # popularity target: surgery_box -> redistribute onto box; else drop+renorm whole
            tgt = redist_to_box(F[k]) if cond=="surgery_box" else zoc(F[k])
            pred = zoc(v)
            pb.append(mre(tgt,pred,box_idx)); pw.append(mre(tgt,pred,whole_idx))
        pb=np.array(pb); pw=np.array(pw)
        return dict(box_mean=float(pb.mean()),box_dev=float(pb.std()),
                    whole_mean=float(pw.mean()),whole_dev=float(pw.std()))

    rows=[]
    for model in ["sp","tp"]:
        for cond in ["none","surgery2","surgery_box"]:
            r=run(model,cond); r.update(model=model,surgery=cond,prior=("same-day" if SAMEDAY else "prev-week"))
            rows.append(r)
            print("  %-3s %-9s | box mean=%.4f dev=%.4f | whole mean=%.4f dev=%.4f"
                  %(model,r["surgery"],r["box_mean"],r["box_dev"],r["whole_mean"],r["whole_dev"]))
    out=dict(event="I-15 Draper crash 2018-09-20, single-link closure (59076), prev-week prior",
             window=[slots20[0],slots20[-1]],n_slots=len(slots20),
             closed_links=sorted(closed),n_closed=len(closed),rows=rows)
    op = OUT_JSON.with_name(OUT_JSON.stem + ("_sameday" if SAMEDAY else "") + ".json")
    op.parent.mkdir(parents=True,exist_ok=True)
    op.write_text(json.dumps(out,indent=2))
    print(f"[done] {time.time()-t0:.0f}s saved {op}")


if __name__=="__main__":
    main()
