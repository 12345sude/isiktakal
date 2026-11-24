import json
import numpy as np
import pandas as pd
import isiktakal as ik

ILCELER = ["Buca", "Karşıyaka", "Bornova", "Konak", "Bayraklı", "Çiğli", "Karabağlar", "Menemen", "Gaziemir", "Narlıdere"]


def veri_hazirla(r, normal_sayisi=350, seed=7):
    rng = np.random.default_rng(seed)
    h = r["hane"]
    secim = h[h.grup != "normal"]
    normal = h[h.grup == "normal"].sample(normal_sayisi, random_state=seed)
    secim = pd.concat([secim, normal]).sort_values("hane")

    d = r["d"]
    s = r["sonuc"]["odeme_kisit"]
    katki = pd.DataFrame(s["katki"][:, :-1], columns=s["oz"])
    katki["hane"], katki["ay"], katki["p"] = d["hane"].values, d["ay"].values, s["p"]
    odeme = r["odeme"].set_index(["hane", "ay"])

    kayitlar = []
    for i, row in enumerate(secim.itertuples()):
        hn = row.hane
        k = katki[katki.hane == hn].set_index("ay")
        risk, nedenler = [], []
        for m in range(12):
            if m in k.index:
                risk.append(round(float(k.loc[m, "p"]), 4))
                nedenler.append([round(float(k.loc[m, f]), 3) for f in s["oz"]])
            else:
                risk.append(None)
                nedenler.append(None)
        o = odeme.loc[hn]
        kes = o[o.kesinti == 1].index
        kayitlar.append(dict(
            id=f"{4100000 + int(hn[1:]) * 7 + 13}",
            ilce=ILCELER[int(rng.integers(0, len(ILCELER)))],
            tip=row.grup,
            g=[round(float(v), 1) for v in r["ga"][hn].values],
            b=[round(float(v), 1) for v in r["ba"][hn].values],
            gec=[int(round(v)) for v in o["gecikme"].values],
            od=[int(v) for v in o["odendi"].values],
            kes=int(kes[0]) if len(kes) else None,
            isr=[bool(v) for v in r["isaret"][hn].values],
            risk=risk, neden=nedenler))
    return dict(esik=round(float(s["esik"]), 4), oz=s["oz"], hane=kayitlar)


if __name__ == "__main__":
    r = ik.calistir(katki=True)
    veri = veri_hazirla(r)
    sablon = open("app/sablon.html", encoding="utf8").read()
    html = sablon.replace("/*VERI*/null", json.dumps(veri, ensure_ascii=False, separators=(",", ":")))
    open("app/index.html", "w", encoding="utf8").write(html)
    print(len(veri["hane"]), "hane", round(len(html) / 1024), "KB")
