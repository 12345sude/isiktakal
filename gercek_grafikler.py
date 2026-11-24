import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gercek_veri as G

plt.rcParams.update({"font.family": "Carlito", "font.size": 10, "axes.grid": True,
                     "grid.color": "#dddddd", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})
AD = {"Affluent": "Varlıklı", "Comfortable": "Orta", "Adversity": "Dezavantajlı"}

oz, s, hane, hava, uyum = G.analiz()
p = G.panel_hazirla().merge(hava[["hdd_gun"]], left_on="month", right_index=True)

fig, ax = plt.subplots(figsize=(3.1, 2.9))
orn = s.sample(4000, random_state=1)
ax.scatter(orn["beklenen"], orn["kwh_gun"], s=3, alpha=0.25, color="tab:blue", linewidths=0)
m = np.percentile(orn["kwh_gun"], 99)
ax.plot([0, m], [0, m], color="tab:red", lw=1)
ax.set_xlim(0, m)
ax.set_ylim(0, m)
ax.set_xlabel("Beklenen (kWh/gün)")
ax.set_ylabel("Gerçekleşen (kWh/gün)")
ax.text(0.04 * m, 0.9 * m, f"Ortanca hata: %{oz['mape_medyan'] * 100:.1f}".replace(".", ","), fontsize=9)
fig.savefig("gsekil1.png")
plt.close()

fig, ax = plt.subplots(figsize=(3.1, 2.9))
gr = ["Affluent", "Comfortable", "Adversity"]
x = np.arange(3)
a = [oz["ilk_oran"][g] * 100 for g in gr]
b = [oz["son_oran"][g] * 100 for g in gr]
c1 = ax.bar(x - 0.18, a, 0.36, color="#bbbbbb", label="İlk kural")
c2 = ax.bar(x + 0.18, b, 0.36, color="tab:blue", label="Yaz kontrolü ile")
ax.bar_label(c1, labels=[f"{v:.1f}".replace(".", ",") for v in a], fontsize=8)
ax.bar_label(c2, labels=[f"{v:.1f}".replace(".", ",") for v in b], fontsize=8)
ax.set_xticks(x, [AD[g] for g in gr])
ax.set_ylabel("İşaretlenen hane (%)")
ax.set_ylim(0, max(a) * 1.3)
ax.legend(fontsize=8, loc="upper right")
fig.savefig("gsekil2.png")
plt.close()

aday = hane[hane.isaret & (hane.r2 >= 0.6) & (hane.egim > 0)].sort_values("r2", ascending=False)
hn = aday.index[0]
g = p[p.lclid == hn].sort_values("month")
u = uyum.set_index("lclid").loc[hn]
g["beklenen"] = np.clip(u.taban + u.egim * g["hdd_gun"], 0.05, None)
etiket = [d.strftime("%m/%y") for d in g["month"]]
xx = np.arange(len(g))
fig, ax = plt.subplots(figsize=(6.3, 2.6))
ax.bar(xx - 0.2, g["beklenen"], 0.4, color="#bbbbbb", label="Beklenen (hava koşuluna göre)")
ax.bar(xx + 0.2, g["kwh_gun"], 0.4, color="tab:blue", label="Gerçekleşen")
sinir = int(np.argmax(g["month"].values >= np.datetime64("2013-07-01")))
ax.axvline(sinir - 0.5, color="black", lw=1, ls=":")
ax.text(sinir - 0.4, ax.get_ylim()[1] * 0.93, "test dönemi", fontsize=9)
ax.set_xticks(xx, etiket, rotation=90, fontsize=8)
ax.set_ylabel("kWh/gün")
ax.legend(fontsize=8.5, loc="upper left")
fig.savefig("gsekil3.png")
plt.close()
open("gsekil3_hane.txt", "w").write(hn)
print(hn, u.r2, u.egim)
