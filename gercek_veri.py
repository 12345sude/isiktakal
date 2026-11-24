import os
import json
import urllib.request
import numpy as np
import pandas as pd
from scipy import stats

KAYNAK = {
    "panel_monthly.parquet": "https://raw.githubusercontent.com/Nasheor/lcl-dtou-causal/main/data/derived/panel_monthly.parquet",
    "weather_hourly_darksky_30minutes.csv": "https://raw.githubusercontent.com/Iavalos/Forecasting-electricity-consumption-in-London/master/weather_hourly_darksky_30minutes.csv",
    "informations_households.csv": "https://raw.githubusercontent.com/Iavalos/Forecasting-electricity-consumption-in-London/master/informations_households.csv",
}
KLASOR = "veri"
HDD_TABAN = 15.5
ISITMA = [1, 2, 3, 10, 11, 12]
OGRENME = ("2012-07-01", "2013-06-01")
TEST = ("2013-07-01", "2013-12-01")


def indir():
    os.makedirs(KLASOR, exist_ok=True)
    for ad, url in KAYNAK.items():
        yol = os.path.join(KLASOR, ad)
        if not os.path.exists(yol):
            urllib.request.urlretrieve(url, yol)


def aylik_hava():
    w = pd.read_csv(os.path.join(KLASOR, "weather_hourly_darksky_30minutes.csv"), usecols=["time", "temperature"])
    w["time"] = pd.to_datetime(w["time"])
    gun = w.set_index("time")["temperature"].resample("D").mean().interpolate()
    hdd = (HDD_TABAN - gun).clip(lower=0)
    ay = pd.DataFrame({"hdd": hdd.resample("MS").sum(), "sicaklik": gun.resample("MS").mean(),
                       "gun": gun.resample("MS").count()})
    ay["hdd_gun"] = ay["hdd"] / ay["gun"]
    return ay


def panel_hazirla():
    p = pd.read_parquet(os.path.join(KLASOR, "panel_monthly.parquet"))
    p = p[(p["group"] == "Std") & (p["halfhours_observed"] >= 0.9 * p["halfhours_expected"])].copy()
    p["gun"] = p["month"].dt.days_in_month
    p["kwh_gun"] = p["kwh_total"] / p["gun"]
    p["donem"] = np.where(p["month"].between(*map(pd.Timestamp, OGRENME)), "ogrenme",
                          np.where(p["month"].between(*map(pd.Timestamp, TEST)), "test", None))
    p = p[p["donem"].notna()]
    sayi = p.groupby(["lclid", "donem"])["month"].count().unstack(fill_value=0)
    yeterli = sayi[(sayi["ogrenme"] >= 11) & (sayi["test"] >= 5)].index
    p = p[p["lclid"].isin(yeterli)]
    bilgi = pd.read_csv(os.path.join(KLASOR, "informations_households.csv"))
    p = p.merge(bilgi[["LCLid", "Acorn_grouped"]], left_on="lclid", right_on="LCLid", how="left")
    return p


def kisit_indeksi(p, hava):
    p = p.merge(hava[["hdd_gun"]], left_on="month", right_index=True)
    sonuc, uyum = [], []
    for h, g in p.groupby("lclid"):
        a = g[g.donem == "ogrenme"]
        b = g[g.donem == "test"].copy()
        x, y = a["hdd_gun"].values, a["kwh_gun"].values
        egim, kesisim, r, _, _ = stats.linregress(x, y)
        b["beklenen"] = np.clip(kesisim + egim * b["hdd_gun"].values, 0.05, None)
        b["egim"] = egim
        uyum.append(dict(lclid=h, r2=r ** 2, egim=egim, taban=kesisim,
                         mape=float(np.mean(np.abs(b["kwh_gun"] - b["beklenen"]) / b["beklenen"]))))
        sonuc.append(b)
    s = pd.concat(sonuc)
    s["ki"] = s["kwh_gun"] / s["beklenen"]
    s["ki_goreli"] = s["ki"] / s.groupby("month")["ki"].transform("median")
    return s, pd.DataFrame(uyum)


def isaretle(s, esik=0.70, bos_kwh=20, yaz_esik=0.85):
    s = s.sort_values(["lclid", "month"]).copy()
    s["alt"] = (s["ki_goreli"] < esik) & (s["kwh_total"] >= bos_kwh) & s["month"].dt.month.isin(ISITMA)
    s["isaret_ilk"] = s.groupby("lclid")["alt"].transform(lambda x: x.astype(int).rolling(2).sum() >= 2)
    yaz = s[s["month"].dt.month.isin([7, 8, 9])].groupby("lclid")["ki_goreli"].median()
    s["yaz_ki"] = s["lclid"].map(yaz)
    s["isaret"] = s["isaret_ilk"] & (s["yaz_ki"] >= yaz_esik)
    return s


def oran_testi(h, sutun):
    t = pd.crosstab(h["grup"], h[sutun])
    return {g: float(h[h.grup == g][sutun].mean()) for g in GRUP}, float(stats.chi2_contingency(t)[1])


GRUP = ["Affluent", "Comfortable", "Adversity"]


def analiz():
    indir()
    hava = aylik_hava()
    p = panel_hazirla()
    s, uyum = kisit_indeksi(p, hava)
    s = isaretle(s)
    ay = s["month"].dt.month
    kis = s[ay.isin([10, 11, 12])].groupby("lclid")["ki_goreli"].median()
    yaz = s[ay.isin([7, 8, 9])].groupby("lclid")["ki_goreli"].median()
    hane = s.groupby("lclid").agg(isaret_ilk=("isaret_ilk", "any"), isaret=("isaret", "any"),
                                  grup=("Acorn_grouped", "first"))
    hane["kis_yaz_farki"] = kis - yaz
    hane["genel_dusus"] = hane["isaret_ilk"] & ~hane["isaret"]
    hane = hane.join(uyum.set_index("lclid"))
    hane = hane[hane.grup.isin(GRUP)]
    isitan = hane[(hane.egim > 0) & (hane.r2 >= 0.5)]
    ilk, p_ilk = oran_testi(hane, "isaret_ilk")
    son, p_son = oran_testi(hane, "isaret")
    gen, p_gen = oran_testi(hane, "genel_dusus")
    isi, p_isi = oran_testi(isitan, "isaret")
    kw = stats.kruskal(*[hane[hane.grup == g].kis_yaz_farki.dropna() for g in GRUP])
    oz = dict(
        hane=int(len(hane)), gozlem=int(len(s)),
        r2_medyan=float(uyum.r2.median()), mape_medyan=float(uyum.mape.median()),
        egim_pozitif=float((uyum.egim > 0).mean()),
        grup_n={g: int((hane.grup == g).sum()) for g in GRUP},
        ilk_sayi=int(hane.isaret_ilk.sum()), ilk_oran=ilk, p_ilk=p_ilk,
        genel_sayi=int(hane.genel_dusus.sum()), genel_oran=gen, p_genel=p_gen,
        son_sayi=int(hane.isaret.sum()), son_oran=son, p_son=p_son,
        isitan_n=int(len(isitan)), isitan_oran=isi, p_isitan=p_isi,
        fark_medyan={g: float(hane[hane.grup == g].kis_yaz_farki.median()) for g in GRUP}, p_fark=float(kw.pvalue),
    )
    return oz, s, hane, hava, uyum


if __name__ == "__main__":
    oz, s, hane, hava, uyum = analiz()
    s.to_csv(os.path.join(KLASOR, "kisit_2013.csv"), index=False)
    json.dump(oz, open(os.path.join(KLASOR, "gercek_ozet.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(oz, ensure_ascii=False, indent=1))
