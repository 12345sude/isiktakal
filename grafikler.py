import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import isiktakal as ik

plt.rcParams.update({"font.family": "Carlito", "font.size": 10, "axes.grid": True,
                     "grid.color": "#dddddd", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})
AY = ["Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"]
AD = {"normal": "Olağan hane", "gizli": "Gizli enerji yoksulu", "kayan": "Kısıtlayıp borca kayan",
      "borclu": "Yalnızca borçlu", "verimlilik": "Verimlilik yatırımı yapan", "bos": "Boş konut / taşınma"}

r = ik.calistir()
h, ki, ga, ba, isaret = r["hane"], r["ki"], r["ga"], r["ba"], r["isaret"]

aday = h[(h.grup == "gizli") & h.baslangic_ay.between(3, 5) & h.isaretli]
hn = aday.hane.iloc[0]
x = np.arange(12)
fig, ax = plt.subplots(figsize=(6.3, 2.6))
ax.bar(x - 0.2, ba[hn].values, 0.4, label="Beklenen (hava koşuluna göre)", color="#bbbbbb")
ax.bar(x + 0.2, ga[hn].values, 0.4, label="Gerçekleşen", color="tab:blue")
ilk = int(np.argmax(isaret[hn].values))
ax.axvline(ilk - 0.5, color="tab:red", lw=1, ls="--")
ax.text(ilk - 0.4, ax.get_ylim()[1] * 0.92, "ilk işaret", color="tab:red", fontsize=9)
ax.set_xticks(x, AY)
ax.set_ylabel("kWh/ay")
ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
fig.savefig("sekil1.png")
plt.close()

sira = ["gizli", "kayan", "verimlilik", "normal", "borclu", "bos"]
deger = ki.rolling(2).max().min()
fig, ax = plt.subplots(figsize=(6.3, 2.8))
ax.boxplot([deger[h[h.grup == g].hane].clip(0, 1.4).values for g in sira],
           widths=0.55, flierprops=dict(markersize=2, alpha=0.4), orientation="horizontal")
ax.set_yticks(range(1, 7), [AD[g] for g in sira])
ax.invert_yaxis()
ax.axvline(0.70, color="tab:red", lw=1, ls="--", label="eşik = 0,70")
ax.set_xlabel("Kısıt İndeksi (art arda iki ayın en düşüğü)")
ax.legend(fontsize=9, loc="lower right")
fig.savefig("sekil2.png")
plt.close()

s = r["sonuc"]
fig, ax = plt.subplots(figsize=(3.1, 2.7))
ax.plot(s["odeme"]["rc"], s["odeme"]["pr"], label=f"Yalnızca ödeme ({s['odeme']['prauc']:.2f})".replace(".", ","), ls="--")
ax.plot(s["odeme_kisit"]["rc"], s["odeme_kisit"]["pr"], label=f"Ödeme + Kısıt İnd. ({s['odeme_kisit']['prauc']:.2f})".replace(".", ","))
ax.set_xlabel("Duyarlılık")
ax.set_ylabel("Kesinlik")
ax.legend(fontsize=8, loc="lower left", title="PR-AUC", title_fontsize=8)
fig.savefig("sekil3.png")
plt.close()

u = r["uyari"]
fig, ax = plt.subplots(figsize=(3.1, 2.7))
gr = ["borclu", "kayan"]
a = [u["odeme"].loc[g, "mean"] for g in gr]
b = [u["odeme_kisit"].loc[g, "mean"] for g in gr]
xx = np.arange(2)
c1 = ax.bar(xx - 0.18, a, 0.36, label="Yalnızca ödeme", color="#bbbbbb")
c2 = ax.bar(xx + 0.18, b, 0.36, label="Ödeme + Kısıt İnd.", color="tab:blue")
ax.bar_label(c1, fmt="%.0f", fontsize=8)
ax.bar_label(c2, fmt="%.0f", fontsize=8)
ax.set_xticks(xx, ["Yalnızca\nborçlu", "Kısıtlayıp\nborca kayan"])
ax.set_ylabel("Kesintiden önceki uyarı süresi (gün)")
ax.set_ylim(0, max(b) * 1.25)
ax.legend(fontsize=8, loc="upper left")
fig.savefig("sekil4.png")
plt.close()

hedef = h.grup.isin(["gizli", "kayan"])
dp, yp, yn = int((h.isaretli & hedef).sum()), int((h.isaretli & ~hedef).sum()), int((~h.isaretli & hedef).sum())
ozet = dict(n={g: int((h.grup == g).sum()) for g in AD}, oran={g: float(h[h.grup == g].isaretli.mean()) for g in AD},
            dp=dp, yp=yp, yn=yn, kesinlik=dp / (dp + yp), duyarlilik=dp / (dp + yn),
            prauc={k: float(v["prauc"]) for k, v in s.items()}, yak5={k: float(v["yakalama5"]) for k, v in s.items()},
            uyari={k: {g: float(u[k].loc[g, "mean"]) for g in gr} for k in u}, nu={g: int(u["odeme"].loc[g, "count"]) for g in gr},
            gozlem=int(len(r["d"])))
json.dump(ozet, open("ozet.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(ozet, ensure_ascii=False))
