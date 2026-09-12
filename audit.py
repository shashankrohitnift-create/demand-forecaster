"""Intense logic audit of sku_engine."""
import numpy as np, pandas as pd, importlib, traceback
import sku_engine as E
importlib.reload(E)
P=F=0
def chk(name, cond, detail=""):
    global P,F
    if cond: P+=1; print(f"  PASS  {name}")
    else:    F+=1; print(f"  FAIL  {name}   {detail}")

def series(vals, start="2024-01-01"):
    return pd.Series(vals, index=pd.date_range(start, periods=len(vals), freq="MS"), dtype=float)

print("\n--- 1. classify() ---")
q,adi,cv2=E.classify(series([10]*12)); chk("constant -> smooth", q=="smooth", q)
chk("constant ADI==1", abs(adi-1)<1e-9, adi)
chk("constant CV2==0", abs(cv2)<1e-9, cv2)
q,_,_=E.classify(series([0,10,0,0,12,0,0,9,0,0,11,0])); chk("gappy -> intermittent/lumpy", q in("intermittent","lumpy"), q)
q,_,_=E.classify(series([0]*12)); chk("all zeros -> sparse", q=="sparse", q)
q,_,_=E.classify(series([5]+[0]*11)); chk("single sale -> sparse", q=="sparse", q)
q,_,_=E.classify(series([10,90,12,88,9,95,11,85,13,92,8,90])); chk("high variance -> erratic", q=="erratic", q)

print("\n--- 2. trim_to_launch() ---")
y,sk=E.trim_to_launch(series([0]*12+[5]*12)); chk("trims 12 leading zeros", sk==12 and len(y)==12, (sk,len(y)))
y,sk=E.trim_to_launch(series([5]*12)); chk("no leading zeros -> no trim", sk==0, sk)
y,sk=E.trim_to_launch(series([0]*20+[5]*3)); chk("too little left -> keeps all", sk==0, sk)
y,sk=E.trim_to_launch(series([0]*12)); chk("all zeros -> no trim", sk==0, sk)
y,sk=E.trim_to_launch(series([5]*10+[0]*5)); chk("trailing zeros KEPT", sk==0 and len(y)==15, (sk,len(y)))

print("\n--- 3. smart_round() ---")
chk("small keeps 2dp", list(E.smart_round([0.56,0.85]))==[0.56,0.85])
chk("mid keeps 1dp", list(E.smart_round([45.34,52.71]))==[45.3,52.7])
chk("large whole", list(E.smart_round([1203.6,980.2]))==[1204.0,980.0])
chk("zeros safe", list(E.smart_round([0,0]))==[0,0])

print("\n--- 4. event_features() (moving seasonality) ---")
idx=pd.date_range("2024-01-01",periods=12,freq="MS")
ev=[("ramadan","2024-03-11","2024-04-09")]
f=E.event_features(idx,ev)
mar=f.loc[pd.Timestamp("2024-03-01"),"ev_ramadan"]; apr=f.loc[pd.Timestamp("2024-04-01"),"ev_ramadan"]
chk("March fraction ~21/31", abs(mar-21/31)<0.02, mar)
chk("April fraction ~9/30", abs(apr-9/30)<0.02, apr)
chk("non-event months zero", f.loc[pd.Timestamp("2024-06-01"),"ev_ramadan"]==0)
chk("fractions <=1", (f<=1).all().all())
f2=E.event_features(idx,[("x","2024-03-01","2024-03-31")])
chk("full month ==1", abs(f2.loc[pd.Timestamp("2024-03-01"),"ev_x"]-1)<1e-9)

print("\n--- 5. fourier_features() ---")
fo=E.fourier_features(idx)
chk("4 cols for K=2", fo.shape[1]==4, fo.shape)
chk("bounded [-1,1]", fo.abs().max().max()<=1.0001)

print("\n--- 6. forecasters ---")
chk("croston flat", len(set(np.round(E.f_croston(series([0,10,0,0,12,0,0,9,0,0,11,0]),6),6)))==1)
chk("croston all-zero -> zeros", (E.f_croston(series([0]*12),6)==0).all())
n=E.f_naive(series(list(range(1,13))),6,blend=0.0); chk("naive(blend=0) repeats last cycle", list(n[:3])==[1.0,2.0,3.0], n[:3])
n=E.f_naive(series(list(range(1,13))),6); chk("naive default blends toward recent level", n[0]>1.0, n[0])
zy=series([0]*8+[0,0,4,5]); chk("naive won't zero out an active SKU", E.f_naive(zy,6)[0]>0, E.f_naive(zy,6)[0])
chk("recent_level uses only recent months", abs(E.recent_level(series([50]*20+[0]*6)))<1e-9, E.recent_level(series([50]*20+[0]*6)))
chk("recent_level of active SKU", abs(E.recent_level(series([0]*10+[6]*6))-6)<1e-9)
n2=E.f_naive(series([5,6,7]),4); chk("naive short -> mean", abs(n2[0]-6)<1e-9, n2[0])
chk("wmape perfect==0", E.wmape([10,20],[10,20])==0)
chk("wmape 10% ", abs(E.wmape([100,100],[90,110])-0.10)<1e-9)
chk("wmape zero-actual -> nan", np.isnan(E.wmape([0,0],[1,1])))

print("\n--- 7. grouping ---")
df=pd.DataFrame({"date":pd.date_range("2024-01-01",periods=6,freq="D").tolist()*2,
                 "sku":["A"]*6+["B"]*6,"rep":["R1","R2"]*6,"qty":[1]*12})
g=E.apply_grouping(df,"sku"); chk("sku mode keys", set(g.series)=={"A","B"})
g=E.apply_grouping(df,"sku_rep"); chk("sku_rep keys", all(E.SEP in s for s in g.series))
g=E.apply_grouping(df.drop(columns=["rep"]),"sku_rep"); chk("no rep col -> falls back to sku", set(g.series)=={"A","B"})
o=pd.DataFrame({"SKU":[f"A{E.SEP}R1"],"WMAPE":[0.1]})
d=E.split_series_key(o,"sku_rep"); chk("split gives 2 cols", d.SKU[0]=="A" and d.Country[0]=="R1", d.to_dict())
d2=E.split_series_key(o,"sku"); chk("sku mode adds no Salesman", "Country" not in d2.columns)

print("\n--- 8. ROP/ROQ maths ---")
out=pd.DataFrame({"SKU":["X"],"Pattern":["smooth"],"Method":["sarimax"],"WMAPE":[0.20],
                  "Jan-26":[300.0],"Feb-26":[300.0],"Mar-26":[600.0]})
fut=pd.date_range("2026-01-01",periods=3,freq="MS")
r=E.compute_rop_roq(out,fut,lead_time_days=14,service_level=95,days_cover=30,days_per_month=30).iloc[0]
avg=400; daily=avg/30; dlt=daily*14; ss=1.65*0.20*dlt
chk("avg monthly", abs(r["Avg monthly demand"]-avg)<1, r["Avg monthly demand"])
chk("daily demand", abs(r["Daily demand"]-daily)<0.2, r["Daily demand"])
chk("lead-time demand", abs(r["Lead-time demand"]-dlt)<1.5, (r["Lead-time demand"],dlt))
chk("safety stock", abs(r["Safety stock"]-ss)<1.5, (r["Safety stock"],ss))
chk("ROP = dlt+ss", abs(r["ROP"]-(dlt+ss))<2, r["ROP"])
chk("ROQ = daily*cover", abs(r["ROQ"]-daily*30)<2, r["ROQ"])
chk("peak ROP > avg ROP", r["ROP at peak month"]>r["ROP"], (r["ROP at peak month"],r["ROP"]))
chk("z(95)=1.65", E.z_for(95)==1.65)
chk("z(99)=2.33", E.z_for(99)==2.33)
chk("z higher service = bigger", E.z_for(99)>E.z_for(80))

print("\n--- 9. end-to-end integrity ---")
df=E.load_long("SAMPLE_sales_data.xlsx")
mat=E.to_monthly_matrix(E.apply_grouping(df,"sku"),"series")
out,fut=E.run(mat,E.DEFAULT_EVENTS,horizon=6,test_months=12)
mc=[d.strftime("%b-%y") for d in fut]
chk("one row per SKU", len(out)==mat.shape[1], (len(out),mat.shape[1]))
chk("no negative forecasts", (out[mc].values>=0).all())
chk("no NaN forecasts", not pd.isna(out[mc].values).any())
chk("horizon cols == 6", len(mc)==6)
chk("Beats_naive consistent", all(
    (r.Beats_naive=="YES")==(r.WMAPE<r.WMAPE_naive) for _,r in out.iterrows()
    if pd.notna(r.WMAPE) and pd.notna(r.WMAPE_naive)))
chk("months <= data length", (out.Months<=mat.shape[0]).all())
chk("forecast starts after last actual", fut[0]>mat.index.max())
# determinism
o2,_=E.run(mat,E.DEFAULT_EVENTS,horizon=6,test_months=12)
chk("deterministic re-run", out[mc].equals(o2[mc]))
# test_months=0
o3,f3=E.run(mat,E.DEFAULT_EVENTS,horizon=6,test_months=0)
chk("test=0 -> no WMAPE", o3.WMAPE.isna().all())
chk("test=0 still forecasts", (o3[[d.strftime('%b-%y') for d in f3]].notna()).all().all())

print("\n--- 10. edge cases ---")
try:
    tiny=pd.DataFrame({"date":pd.date_range("2025-01-01",periods=60,freq="D"),
                       "sku":["Z"]*60,"qty":[3]*60})
    m=E.to_monthly_matrix(E.apply_grouping(tiny,"sku"),"series")
    o,_=E.run(m,E.DEFAULT_EVENTS,horizon=3,test_months=1)
    chk("2-month history runs", len(o)==1, )
except Exception as ex: chk("2-month history runs", False, str(ex)[:60])
try:
    one=pd.DataFrame({"date":[pd.Timestamp("2025-01-05")],"sku":["Q"],"qty":[7]})
    m=E.to_monthly_matrix(E.apply_grouping(one,"sku"),"series")
    o,_=E.run(m,E.DEFAULT_EVENTS,horizon=3,test_months=0)
    chk("single row survives", len(o)==1)
except Exception as ex: chk("single row survives", False, str(ex)[:60])
try:
    o,_=E.run(mat,[],horizon=3,test_months=6)   # no events
    chk("empty event list ok", len(o)==mat.shape[1])
except Exception as ex: chk("empty event list ok", False, str(ex)[:60])
try:
    neg=pd.DataFrame({"date":pd.date_range("2024-01-01",periods=400,freq="D"),
                      "sku":["N"]*400,"qty":[5,-2]*200})
    m=E.to_monthly_matrix(E.apply_grouping(neg,"sku"),"series")
    o,f=E.run(m,E.DEFAULT_EVENTS,horizon=3,test_months=3)
    chk("returns/negatives don't crash", len(o)==1)
    chk("negatives clipped to >=0", (o[[d.strftime('%b-%y') for d in f]].values>=0).all())
except Exception as ex: chk("returns/negatives", False, str(ex)[:60])

print(f"\n{'='*46}\n  PASSED {P}   FAILED {F}\n{'='*46}")
