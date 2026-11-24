import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import GroupKFold

SEED = 42
N = 4000
GUNLER = pd.date_range("2024-01-01", "2025-12-31", freq="D")
HDD_TABAN, CDD_TABAN = 15.0, 22.0

GRUPLAR = {
    "normal": 0.81,
    "gizli": 0.06,
    "kayan": 0.03,
    "borclu": 0.05,
    "verimlilik": 0.03,
    "bos": 0.02,
}

ODEME = ["gecikme", "gecikme_ort3", "odenmemis_3"]
KISIT = ["ki", "ki_min3", "ki_alti_08", "ki_trend"]


def sicaklik(gunler, rng):
    gun = gunler.dayofyear.values
    mevsim = 18.5 - 9.5 * np.cos(2 * np.pi * (gun - 20) / 365.25)
    hava = np.zeros(len(gunler))
    for i in range(1, len(gunler)):
        hava[i] = 0.75 * hava[i - 1] + rng.normal(0, 1.6)
    return mevsim + hava


def sentetik_veri(n=N, seed=SEED):
    rng = np.random.default_rng(seed)
    t = sicaklik(GUNLER, rng)
    hdd = np.clip(HDD_TABAN - t, 0, None)
    cdd = np.clip(t - CDD_TABAN, 0, None)
    hava = pd.DataFrame({"tarih": GUNLER, "sicaklik": t, "hdd": hdd, "cdd": cdd})

    grup = rng.choice(list(GRUPLAR), size=n, p=list(GRUPLAR.values()))
    taban = rng.lognormal(np.log(4.5), 0.35, n)
    isitma = rng.uniform(0.05, 0.45, n)
    sogutma = rng.uniform(0.10, 0.55, n)
    baslangic = rng.integers(0, 11, n)
    yasam = rng.uniform(0.85, 1.15, n)
    dusus = (grup == "normal") & (rng.random(n) < 0.06)
    yasam[dusus] = rng.uniform(0.62, 0.85, dusus.sum())

    ay25 = (GUNLER.year == 2025) * (GUNLER.month - 1)
    yil25 = GUNLER.year == 2025
    satirlar = []
    for h in range(n):
        beklenen = taban[h] + isitma[h] * hdd + sogutma[h] * cdd
        y = beklenen * rng.lognormal(0, 0.20, len(GUNLER))
        y = np.where(yil25, y * yasam[h], y)
        aktif = yil25 & (ay25 >= baslangic[h])
        if grup[h] in ("gizli", "kayan"):
            k = rng.uniform(0.45, 0.85)
            kisitli = (taban[h] * k + isitma[h] * 0.15 * hdd + sogutma[h] * 0.3 * cdd)
            y = np.where(aktif, kisitli * rng.lognormal(0, 0.20, len(GUNLER)), y)
        elif grup[h] == "verimlilik":
            y = np.where(aktif, y * rng.uniform(0.70, 0.88), y)
        elif grup[h] == "bos":
            y = np.where(aktif, rng.uniform(0.15, 0.45, len(GUNLER)), y)
        satirlar.append(y)

    tuketim = pd.DataFrame(np.array(satirlar).T, index=GUNLER)
    tuketim.columns = [f"H{h:04d}" for h in range(n)]
    hane = pd.DataFrame({"hane": tuketim.columns, "grup": grup, "baslangic_ay": baslangic})

    odeme = []
    for h in range(n):
        seri, kesildi = 0, False
        for m in range(12):
            gec = rng.exponential(3) if rng.random() < 0.85 else rng.uniform(10, 35)
            odendi = rng.random() > 0.02
            if grup[h] == "borclu" and m >= baslangic[h]:
                k = m - baslangic[h]
                gec = 10 + 12 * k + rng.normal(0, 4)
                odendi = rng.random() > 0.25 + 0.2 * k
            if grup[h] == "kayan" and m >= baslangic[h] + 2:
                k = m - baslangic[h] - 2
                gec = 15 + 12 * k + rng.normal(0, 4)
                odendi = rng.random() > 0.35 + 0.2 * k
            if grup[h] == "gizli":
                gec, odendi = rng.exponential(2), True
            seri = 0 if odendi else seri + 1
            olay = (not kesildi) and seri >= 3
            kesildi = kesildi or olay
            odeme.append(dict(hane=tuketim.columns[h], ay=m, gecikme=max(gec, 0),
                              odendi=int(odendi), kesinti=int(olay)))
    return hava, tuketim, hane, pd.DataFrame(odeme)


def kisit_indeksi(hava, tuketim):
    w = hava.set_index("tarih")
    x24 = w.loc["2024", ["hdd", "cdd"]].values
    x25 = w.loc["2025", ["hdd", "cdd"]].values
    gercek = tuketim.loc["2025"]
    beklenen = pd.DataFrame(index=gercek.index, columns=tuketim.columns, dtype=float)
    for c in tuketim.columns:
        m = LinearRegression().fit(x24, tuketim.loc["2024", c].values)
        beklenen[c] = np.clip(m.predict(x25), 0.1, None)
    ga = gercek.resample("MS").sum()
    ba = beklenen.resample("MS").sum()
    return ga / ba, ga, ba


def isaretle(ki, ga, esik=0.70, ardisik=2, bos_kwh=20):
    alt = (ki < esik).astype(int).rolling(ardisik).sum() >= ardisik
    bos = (ga < bos_kwh).rolling(ardisik).max().astype(bool)
    return alt & ~bos


def panel(ki, odeme, ufuk=3):
    k = ki.reset_index(drop=True).melt(ignore_index=False, var_name="hane", value_name="ki")
    k["ay"] = k.index
    d = odeme.merge(k.reset_index(drop=True), on=["hane", "ay"]).sort_values(["hane", "ay"])
    g = d.groupby("hane")
    d["gecikme_ort3"] = g["gecikme"].transform(lambda s: s.rolling(3, 1).mean())
    d["odenmemis_3"] = g["odendi"].transform(lambda s: (1 - s).rolling(3, 1).sum())
    d["ki_min3"] = g["ki"].transform(lambda s: s.rolling(3, 1).min())
    d["ki_alti_08"] = g["ki"].transform(lambda s: (s < 0.8).rolling(3, 1).sum())
    d["ki_trend"] = g["ki"].transform(lambda s: s.diff(2)).fillna(0)
    d["hedef"] = g["kesinti"].transform(
        lambda s: sum(s.shift(-i).fillna(0) for i in range(1, ufuk + 1))).clip(0, 1)
    d["kesildi"] = g["kesinti"].cumsum()
    kay = d[d.kesinti == 1].set_index("hane")["ay"]
    d["kesinti_ay"] = d["hane"].map(kay)
    return d[d["kesildi"] == 0].reset_index(drop=True)


def model_kur():
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                              min_child_samples=30, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.9, random_state=SEED, verbose=-1,
                              importance_type="gain")


def risk_modeli(d, kat=5, katki=False):
    sonuc = {}
    for ad, oz in [("odeme", ODEME), ("odeme_kisit", ODEME + KISIT)]:
        p = np.zeros(len(d))
        kt = np.zeros((len(d), len(oz) + 1))
        for tr, te in GroupKFold(kat).split(d, groups=d["hane"]):
            m = model_kur().fit(d.iloc[tr][oz], d.iloc[tr]["hedef"])
            p[te] = m.predict_proba(d.iloc[te][oz])[:, 1]
            if katki:
                kt[te] = m.predict(d.iloc[te][oz], pred_contrib=True)
        y = d["hedef"].values
        esik = np.quantile(p, 0.95)
        pr, rc, _ = precision_recall_curve(y, p)
        sonuc[ad] = dict(oz=oz, p=p, katki=kt, esik=esik, pr=pr, rc=rc,
                         prauc=average_precision_score(y, p),
                         yakalama5=(p[y == 1] >= esik).mean())
    return sonuc


def erken_uyari(d, sonuc, hane):
    grup = hane.set_index("hane")["grup"]
    tablo = {}
    for ad, r in sonuc.items():
        x = d.assign(p=r["p"], grup=d["hane"].map(grup))
        satir = []
        for h, s in x.groupby("hane"):
            if s["kesinti_ay"].isna().all():
                continue
            ka = s["kesinti_ay"].iloc[0]
            ilk = s[(s.p >= r["esik"]) & (s.ay < ka)]
            satir.append((s["grup"].iloc[0], (ka - ilk.ay.min()) * 30 if len(ilk) else 0))
        tablo[ad] = pd.DataFrame(satir, columns=["grup", "gun"]).groupby("grup")["gun"].agg(["mean", "count"])
    return tablo


def calistir(katki=False):
    hava, tuketim, hane, odeme = sentetik_veri()
    ki, ga, ba = kisit_indeksi(hava, tuketim)
    isaret = isaretle(ki, ga)
    hane["isaretli"] = hane["hane"].map(isaret.any())
    d = panel(ki, odeme)
    sonuc = risk_modeli(d, katki=katki)
    return dict(hava=hava, tuketim=tuketim, hane=hane, odeme=odeme, ki=ki, ga=ga, ba=ba,
                isaret=isaret, d=d, sonuc=sonuc, uyari=erken_uyari(d, sonuc, hane))


if __name__ == "__main__":
    r = calistir()
    h = r["hane"]
    hedef = h.grup.isin(["gizli", "kayan"])
    dp = (h.isaretli & hedef).sum()
    yp = (h.isaretli & ~hedef).sum()
    yn = (~h.isaretli & hedef).sum()
    print(f"Kisit tespiti: kesinlik {dp / (dp + yp):.2f}, duyarlilik {dp / (dp + yn):.2f}")
    print(h.groupby("grup")["isaretli"].mean().round(3).to_string())
    print()
    for ad, s in r["sonuc"].items():
        print(f"{ad:12s} PR-AUC {s['prauc']:.3f}  ilk %5 yakalama {s['yakalama5']:.2f}")
    print()
    for ad, t in r["uyari"].items():
        print(ad)
        print(t.round(0).to_string())
