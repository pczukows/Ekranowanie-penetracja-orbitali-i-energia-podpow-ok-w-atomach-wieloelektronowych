import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Od ekranowania do energii podpowłok", page_icon="⚛️", layout="wide")

RY = 13.605693

# Konfiguracje referencyjne atomów obojętnych Z=1-30. Dla Cr i Cu użyto
# konfiguracji stanów podstawowych, a nie mechanicznego wyniku reguły n+l.
ELEMENTS = {
    "H":  (1,  "wodór",      {"1s":1}),
    "He": (2,  "hel",        {"1s":2}),
    "Li": (3,  "lit",        {"1s":2,"2s":1}),
    "Be": (4,  "beryl",      {"1s":2,"2s":2}),
    "B":  (5,  "bor",        {"1s":2,"2s":2,"2p":1}),
    "C":  (6,  "węgiel",     {"1s":2,"2s":2,"2p":2}),
    "N":  (7,  "azot",       {"1s":2,"2s":2,"2p":3}),
    "O":  (8,  "tlen",       {"1s":2,"2s":2,"2p":4}),
    "F":  (9,  "fluor",      {"1s":2,"2s":2,"2p":5}),
    "Ne": (10, "neon",       {"1s":2,"2s":2,"2p":6}),
    "Na": (11, "sód",        {"1s":2,"2s":2,"2p":6,"3s":1}),
    "Mg": (12, "magnez",     {"1s":2,"2s":2,"2p":6,"3s":2}),
    "Al": (13, "glin",       {"1s":2,"2s":2,"2p":6,"3s":2,"3p":1}),
    "Si": (14, "krzem",      {"1s":2,"2s":2,"2p":6,"3s":2,"3p":2}),
    "P":  (15, "fosfor",     {"1s":2,"2s":2,"2p":6,"3s":2,"3p":3}),
    "S":  (16, "siarka",     {"1s":2,"2s":2,"2p":6,"3s":2,"3p":4}),
    "Cl": (17, "chlor",      {"1s":2,"2s":2,"2p":6,"3s":2,"3p":5}),
    "Ar": (18, "argon",      {"1s":2,"2s":2,"2p":6,"3s":2,"3p":6}),
    "K":  (19, "potas",      {"1s":2,"2s":2,"2p":6,"3s":2,"3p":6,"4s":1}),
    "Ca": (20, "wapń",       {"1s":2,"2s":2,"2p":6,"3s":2,"3p":6,"4s":2}),
    "Sc": (21, "skand",      {"1s":2,"2s":2,"2p":6,"3s":2,"3p":6,"3d":1,"4s":2}),
    "Ti": (22, "tytan",      {"1s":2,"2s":2,"2p":6,"3s":2,"3p":6,"3d":2,"4s":2}),
    "V":  (23, "wanad",      {"1s":2,"2s":2,"2p":6,"3s":2,"3p":6,"3d":3,"4s":2}),
    "Cr": (24, "chrom",      {"1s":2,"2s":2,"2p":6,"3s":2,"3p":6,"3d":5,"4s":1}),
    "Mn": (25, "mangan",     {"1s":2,"2s":2,"2p":6,"3s":2,"3p":6,"3d":5,"4s":2}),
    "Fe": (26, "żelazo",     {"1s":2,"2s":2,"2p":6,"3s":2,"3p":6,"3d":6,"4s":2}),
    "Co": (27, "kobalt",     {"1s":2,"2s":2,"2p":6,"3s":2,"3p":6,"3d":7,"4s":2}),
    "Ni": (28, "nikiel",     {"1s":2,"2s":2,"2p":6,"3s":2,"3p":6,"3d":8,"4s":2}),
    "Cu": (29, "miedź",      {"1s":2,"2s":2,"2p":6,"3s":2,"3p":6,"3d":10,"4s":1}),
    "Zn": (30, "cynk",       {"1s":2,"2s":2,"2p":6,"3s":2,"3p":6,"3d":10,"4s":2}),
}

ORBITAL_ORDER = ["1s","2s","2p","3s","3p","4s","3d","4p","5s","4d","5p","6s","4f","5d","6p","7s","5f","6d","7p"]
GROUP_ORDER = [(1,"sp"),(2,"sp"),(3,"sp"),(3,"d"),(4,"sp"),(4,"d"),(4,"f"),(5,"sp"),(5,"d"),(5,"f"),(6,"sp"),(6,"d"),(7,"sp")]

HELP = {
    "zeff": "Efektywny ładunek jądra Z_eff jest parametrem opisującym średnie przyciąganie elektronu przez jądro po uwzględnieniu ekranowania. Nie jest nowym rzeczywistym ładunkiem jądra.",
    "shielding": "Ekranowanie wynika z odpychania elektron-elektron i przestrzennego rozkładu gęstości. Zwrot „elektrony zasłaniają jądro” jest tylko przybliżonym opisem.",
    "penetration": "Penetracja oznacza obecność danego orbitalu w obszarze blisko jądra. Przy porównywalnym n orbitale s penetrują zwykle silniej niż p, a p silniej niż d.",
    "slater": "Reguły Slatera są półempirycznym przybliżeniem Z_eff. Ułatwiają porównania, lecz nie zastępują dokładnych bliczeń danych eksperymentalnych (spektroskopowych).",
}


def integrate_curve(y, x):
    return np.trapezoid(y, x)


def rgba(hex_color, alpha):
    value=hex_color.lstrip("#")
    r,g,b=(int(value[i:i+2],16) for i in (0,2,4))
    return f"rgba({r},{g},{b},{alpha})"


def orbital_nl(orbital):
    return int(orbital[0]), "spdf".index(orbital[1])


def config_string(config):
    return " ".join(f"{orb}{count}" for orb,count in config.items() if count)


def intro(title, text, goals):
    st.header(title)
    st.info(text)
    st.markdown("**Po pracy z panelem student powinien:**")
    st.markdown("\n".join(f"- {g}" for g in goals))


def generalized_laguerre(k, alpha, x):
    # L_k^alpha(x), wystarcza dla małych liczb kwantowych użytych w aplikacji.
    total = np.zeros_like(x, dtype=float)
    for m in range(k + 1):
        coeff = ((-1)**m) * math.comb(k + alpha, k - m) / math.factorial(m)
        total += coeff * x**m
    return total


def hydrogenic_radial(n, l, r, z=1.0):
    rho = 2*z*r/n
    pref = math.sqrt((2*z/n)**3 * math.factorial(n-l-1)/(2*n*math.factorial(n+l)))
    return pref * np.exp(-rho/2) * rho**l * generalized_laguerre(n-l-1, 2*l+1, rho)


def slater_zeff(Z, config, target):
    n, l = orbital_nl(target)
    target_count = config.get(target, 0)
    if target_count < 1:
        return None
    same = 0
    n_minus_1 = 0
    lower = 0
    contributions = []
    if l <= 1:  # ns lub np
        same = sum(v for orb,v in config.items() if orbital_nl(orb)[0] == n and orbital_nl(orb)[1] <= 1) - 1
        n_minus_1 = sum(v for orb,v in config.items() if orbital_nl(orb)[0] == n-1)
        lower = sum(v for orb,v in config.items() if orbital_nl(orb)[0] <= n-2)
        same_factor = 0.30 if n == 1 else 0.35
        S = same_factor*same + 0.85*n_minus_1 + lower
        contributions = [("ta sama grupa (ns,np)",same,same_factor,same_factor*same),("powłoka n−1",n_minus_1,0.85,0.85*n_minus_1),("powłoki n−2 i niższe",lower,1.00,lower)]
    else:  # nd lub nf
        same = config.get(target,0)-1
        lower = 0
        for orb,v in config.items():
            on, ol = orbital_nl(orb)
            if on < n or (on == n and ol < l):
                lower += v
        S = 0.35*same + lower
        contributions = [("ta sama grupa nd/nf",same,0.35,0.35*same),("wszystkie grupy po lewej",lower,1.00,lower)]
    return {"S":S,"Zeff":Z-S,"contributions":contributions}


def available_levels(max_n):
    return [f"{n}{'spdf'[l]}" for n in range(1,max_n+1) for l in range(min(n,4))]


def split_level_values(max_n=4, split_strength=0.0):
    levels=available_levels(max_n);values={}
    for orb in levels:
        n,l=orbital_nl(orb)
        values[orb]=-1/n**2+split_strength*{0:-0.12,1:-0.055,2:-0.010,3:0.0}.get(l,0)/n
    return levels,values


def energy_diagram(max_n=4, split_strength=0.0):
    levels,values=split_level_values(max_n,split_strength)
    palette={"s":"#38bdf8","p":"#22c55e","d":"#f97316","f":"#a78bfa"}
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.06,row_heights=[.80,.20])
    upper=[o for o in levels if o!="1s"];xmax=max(2,len(upper)+1)
    for i,orb in enumerate(upper,start=1):
        y=values[orb];color=palette[orb[1]]
        fig.add_shape(type="line",x0=.25,x1=xmax-.25,y0=y,y1=y,line=dict(color=color,width=1.5,dash="dash"),row=1,col=1)
        fig.add_trace(go.Scatter(x=[i-.27,i+.27],y=[y,y],mode="lines+text",line=dict(color=color,width=6),text=[orb,""],textposition="top center",showlegend=False,hovertemplate=f"{orb}<br>energia względna={y:.4f}<extra></extra>"),row=1,col=1)
    y1=values["1s"];color=palette["s"]
    fig.add_shape(type="line",x0=.25,x1=xmax-.25,y0=y1,y1=y1,line=dict(color=color,width=1.5,dash="dash"),row=2,col=1)
    fig.add_trace(go.Scatter(x=[.73,1.27],y=[y1,y1],mode="lines+text",line=dict(color=color,width=6),text=["1s",""],textposition="top center",showlegend=False),row=2,col=1)
    vals=[values[o] for o in upper];pad=max(.018,(max(vals)-min(vals))*.15)
    fig.update_yaxes(range=[min(vals)-pad,max(vals)+pad],title="energia względna",gridcolor="#263449",row=1,col=1)
    fig.update_yaxes(range=[y1-.06,y1+.06],title="1s",gridcolor="#263449",row=2,col=1)
    fig.update_xaxes(visible=False,range=[0,xmax],row=1,col=1);fig.update_xaxes(visible=False,range=[0,xmax],row=2,col=1)
    fig.update_layout(height=610,title=f"Poziomy energetyczne dla n=1…{max_n}",paper_bgcolor="#07111f",plot_bgcolor="#07111f",font=dict(color="#e5e7eb"),margin=dict(l=48,r=20,t=55,b=20))
    fig.add_annotation(x=.01,y=.205,xref="paper",yref="paper",text="//",showarrow=False,font=dict(size=20,color="#94a3b8"))
    return fig


def radial_shape_figure(n=2, zoom=3.0):
    """Pokazuje R(r) i |R(r)|² bez obliczania pola powierzchni od r=0 do r_cut."""
    r=np.linspace(0,max(12,4*n*n),3000);colors=["#38bdf8","#22c55e","#f97316","#a78bfa"]
    fig=make_subplots(rows=1,cols=2,subplot_titles=("Funkcja radialna R(r): znak, węzły i płaty","Gęstość radialna |R(r)|² — powiększenie przy jądrze"))
    rows=[]
    for l in range(min(n,4)):
        letter='spdf'[l];color=colors[l];R=hydrogenic_radial(n,l,r)
        Rn=R/max(abs(R).max(),1e-15);D=R*R;Dn=D/max(D.max(),1e-15)
        mask=r<=zoom
        fig.add_trace(go.Scatter(x=r,y=Rn,mode="lines",line=dict(color=color,width=3),name=f"{n}{letter} — R",legendgroup=f"{n}{letter}"),row=1,col=1)
        fig.add_trace(go.Scatter(x=r[mask],y=Dn[mask],mode="lines",line=dict(color=color,width=3),name=f"{n}{letter} — |R|²",legendgroup=f"{n}{letter}",showlegend=False),row=1,col=2)
        rows.append((f"{n}{letter}",float(Dn[0]),"niezerowa" if abs(R[0])>1e-10 else "zerowa"))
    fig.add_hline(y=0,line_color="#94a3b8",line_width=1,row=1,col=1)
    fig.update_xaxes(title="r/a₀",row=1,col=1);fig.update_xaxes(title=f"r/a₀ (0–{zoom:.1f})",range=[0,zoom],row=1,col=2)
    fig.update_yaxes(title="R(r), skala względna",row=1,col=1);fig.update_yaxes(title="|R(r)|², skala względna",row=1,col=2)
    fig.update_layout(height=500,paper_bgcolor="#07111f",plot_bgcolor="#07111f",font=dict(color="#e5e7eb"),legend=dict(orientation="h",y=1.13),margin=dict(l=30,r=20,t=70,b=25))
    return fig,rows

st.title("⚛️ Od ekranowania do energii podpowłok")
st.caption("Odpychanie elektronów → ekranowanie → efektywny ładunek jądra → penetracja → energia podpowłok.")
st.warning("Aplikacja wyjaśnia przyczyny zmian energii i stosuje przybliżenie Slatera. Nie generuje jeszcze pełnych konfiguracji wszystkich pierwiastków.")

t1,t2,t3,t4,t5 = st.tabs([
    "1. Poziomy energetyczne",
    "2. Ekranowanie i Z_eff",
    "3. Penetracja orbitali",
    "4. Problem 4s–3d",
    "5. Ćwiczenia",
])

with t1:
    intro("Od degeneracji poziomów energetycznych do rozszczepienia powłok","W modelu wodoropodobnym energia zależy od n, więc podpowłoki o tym samym n mają tę samą energię. W atomie wieloelektronowym oddziaływania elektronowe i różne kształty funkcji radialnych usuwają tę degenerację.",["wyjaśniać degenerację poziomów energetycznych","rozpoznawać wpływ l w atomie wieloelektronowym","odróżniać kształt funkcji radialnej od radialnego prawdopodobieństwa w powłoce"])
    c1,c2=st.columns(2)
    max_n=c1.select_slider("Zakres poziomów n",options=[2,3,4,5],value=2,format_func=lambda v:f"n=1…{v}")
    split=c2.slider("Siła rozszczepienia",0.0,2.0,0.0,0.1,help="Wartość 0 pokazuje degenerację zależną tylko od n.")
    st.plotly_chart(energy_diagram(max_n,split),use_container_width=True,config={"displaylogo":False})
    st.caption("Kolorowe przerywane linie ułatwiają porównanie energii. Przerwa na osi energii poprawia czytelność różnic poziomów wyższych.")

    st.subheader("Kształt funkcji radialnej a penetracja")
    st.warning("Zwracamy uwagę na gęstość radialną przy jądrze- prawy panel")
    r1,r2=st.columns(2)
    radial_n=r1.select_slider("n dla porównania radialnego",options=[2,3,4,5],value=2)
    zoom=r2.slider("Zakres powiększenia przy jądrze [a₀]",0.5,5.0,1.5,0.25)
    radial_fig,radial_rows=radial_shape_figure(radial_n,zoom)
    st.plotly_chart(radial_fig,use_container_width=True,config={"displaylogo":False})
    st.dataframe(pd.DataFrame(radial_rows,columns=["podpowłoka","względne |R(0)|²","R(0)"]),use_container_width=True,hide_index=True)
    st.info("O penetracji decyduje wewnętrzna struktura całej funkcji radialnej: jej zachowanie przy jądrze, obecność wewnętrznych części i węzłów oraz nakładanie z obszarem zajmowanym przez elektrony rdzenia. Dla wspólnego n orbitale s penetrują zwykle silniej niż p, a p silniej niż d.")
    st.success("Dla n=2 funkcja 2s ma niezerową wartość przy jądrze i wewnętrzny płat oddzielony węzłem, natomiast 2p znika przy r=0. Dlatego 2s jest bardziej penetrujący i w atomie wieloelektronowym zwykle niższy energetycznie niż 2p.")
    st.caption("Wykres |R|² jest znormalizowany osobno dla każdego orbitalu i służy do porównania kształtu blisko jądra. Nie wyznacza samodzielnie ilościowej energii podpowłoki.")
    q=st.radio("Stany 3s, 3p i 3d:",["mają różne energie z powodu ekranowania","mają tę samą energię, bo energia zależy od n","nie istnieją równocześnie"],index=None,key="quiz_h")
    if q:st.success("Poprawnie." if q.startswith("mają tę samą") else "W atomie jednoelektronowym nie ma ekranowania przez inne elektrony; energia zależy od n.")

with t2:
    intro("Ekranowanie i efektywny ładunek jądra","Reguły Slatera pozwalają półempirycznie oszacować stałą ekranowania S oraz Z_eff=Z−S dla wybranego elektronu. Wynik zależy od tego, który elektron analizujemy.",["obliczać Z_eff według Slatera","wyjaśniać różne ekranowanie elektronów","wskazywać ograniczenia przybliżenia"])
    c1,c2=st.columns(2)
    label=c1.selectbox("Atom",list(ELEMENTS),index=list(ELEMENTS).index("Fe"),format_func=lambda s:f"{s} — {ELEMENTS[s][1]}",help="Zakres Z=1–30 zawiera referencyjne konfiguracje podstawowe; Cr i Cu nie są generowane mechaniczną regułą.")
    Z,name,config=ELEMENTS[label]
    occupied=[orb for orb in ORBITAL_ORDER if config.get(orb,0)>0]
    target=c2.selectbox("Elektron analizowany w podpowłoce",occupied,index=len(occupied)-1,help="Wybieramy jeden elektron z zajętej podpowłoki. Pozostałe elektrony tworzą wkład do S.")
    result=slater_zeff(Z,config,target)
    st.code(f"{label}: {config_string(config)}")
    m=st.columns(3) # zmieniłem z 4 na 3, bo nie wyświetlamy już proxy
    m[0].metric("Ładunek jądra Z",Z)
    m[1].metric("Stała ekranowania S",f"{result['S']:.2f}",help=HELP["shielding"])
    m[2].metric("Efektywny ładunek Z_eff",f"{result['Zeff']:.2f}",help=HELP["zeff"])
    n_target,_=orbital_nl(target)
    # proxy=-RY*result['Zeff']**2/n_target**2
    # m[3].metric("Wskaźnik stabilizacji",f"{proxy:.2f} eV",help=HELP["proxy"])
    rows=[{"grupa elektronów":desc,"liczba":count,"współczynnik":factor,"wkład do S":value} for desc,count,factor,value in result["contributions"]]
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    st.markdown("### Interpretacja")
    st.info(f"Dla elektronu {target} w {label} reguły Slatera dają S={result['S']:.2f}, więc Z_eff≈{result['Zeff']:.2f}. Oznacza to średnio słabsze przyciąganie niż przez nieekranowane jądro +{Z}e.")
    

with t3:
    intro("Penetracja orbitali","Penetracja opisuje obecność gęstości elektronowej blisko jądra. Porównujemy dokładne radialne funkcje wodoropodobne dla wspólnego n.",["czytać rozkład radialny","porównywać s, p i d przy tym samym n","odróżniać penetrację od średniego promienia"])
    n_pen=st.slider("Wspólna główna liczba kwantowa n",2,5,3,help="Dla danego n dostępne są l=0,...,n−1.")
    r_cut=st.slider("Promień obszaru wewnętrznego r_cut [a₀]",0.2,8.0,2.0,0.2,help="Aplikacja oblicza prawdopodobieństwo znalezienia elektronu wewnątrz tej sfery.")
    r=np.linspace(0,35,2500)
    fig=go.Figure(); rows=[]; colors=["#38bdf8","#22c55e","#f97316","#a78bfa"]
    for l in range(min(n_pen,4)):
        letter="spdf"[l]
        R=hydrogenic_radial(n_pen,l,r)
        P=r*r*R*R
        norm=integrate_curve(P,r); P=P/norm
        inside=integrate_curve(P[r<=r_cut],r[r<=r_cut])
        mean=integrate_curve(r*P,r)
        rows.append({"podpowłoka":f"{n_pen}{letter}","l":l,f"P(r<{r_cut:.1f}a₀)":inside,"<r> [a₀]":mean})
        fig.add_trace(go.Scatter(x=r,y=P,mode="lines",line=dict(width=3,color=colors[l]),name=f"{n_pen}{letter}"))
    fig.add_vline(x=r_cut,line_dash="dot",line_color="#facc15",annotation_text="r_cut")
    fig.update_layout(height=520,xaxis_title="r/a₀",yaxis_title="znormalizowane prawdopodobieństwo radialne",paper_bgcolor="#07111f",plot_bgcolor="#07111f",font=dict(color="#e5e7eb"),legend=dict(orientation="h",y=1.08),margin=dict(l=20,r=20,t=40,b=20))
    st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
    st.dataframe(pd.DataFrame(rows).style.format({f"P(r<{r_cut:.1f}a₀)":"{:.4f}","<r> [a₀]":"{:.3f}"}),use_container_width=True,hide_index=True)
    st.info("Większa wartość P(r<r_cut) oznacza większy udział gęstości w obszarze wewnętrznym. Orbital może mieć jednocześnie zewnętrzne maksimum i niewielki, lecz ważną część penetrującą blisko jądra.")
    st.warning("O kolejności penetracji decyduje zachowanie całej funkcji radialnej, szczególnie jej część wewnętrzna.")
    st.info("Ekranowanie i penetracja wyjaśniają jakościowo, dlaczego energia w atomach wieloelektronowych zależy także od l. Nie wyznaczają jednak uniwersalnej kolejności wszystkich podpowłok; decyduje energia całego stanu i dane referencyjne.")

with t4:
    intro("Dlaczego relacja 4s–3d nie jest prostą regułą?","W K i Ca podpowłoka 4s jest obsadzana przed 3d. Po rozpoczęciu obsadzania 3d względne energie zmieniają się, a przy tworzeniu kationów metali przejściowych elektrony 4s są zwykle usuwane przed 3d.",["odróżniać kolejność obsadzania od kolejności jonizacji","wyjaśniać zmienność energii z obsadzeniem","unikać bezwzględnego diagramu 4s<3d"])
    scenario=st.selectbox("Scenariusz",["K i Ca — początek 4s","Sc–Zn — obsadzanie 3d","Fe → Fe²⁺ → Fe³⁺","Cr i Cu — zapowiedź wyjątków"],help="Scenariusze są jakościowe i przygotowują do osobnej aplikacji o pełnych konfiguracjach oraz wszystkich wyjątkach.")
    if scenario.startswith("K"):
        st.code("K:  [Ar] 4s1\nCa: [Ar] 4s2")
        st.info("Dla neutralnych K i Ca dodanie elektronu do 4s daje stan podstawowy przed rozpoczęciem szeregu 3d.")
    elif scenario.startswith("Sc"):
        st.code("Sc: [Ar] 3d1 4s2  →  ...  →  Zn: [Ar] 3d10 4s2")
        st.info("Po pojawieniu się elektronów 3d ich przyciąganie i ekranowanie zmieniają energię całego układu. Nie należy traktować energii 4s i 3d jako dwóch nieruchomych linii.")
    elif scenario.startswith("Fe"):
        st.code("Fe:    [Ar] 3d6 4s2\nFe2+:  [Ar] 3d6\nFe3+:  [Ar] 3d5")
        st.success("Przy tworzeniu kationów Fe najpierw usuwane są elektrony 4s. Kolejność usuwania nie jest prostym odwróceniem szkolnej kolejności zapełniania.")
    else:
        st.code("proste przewidywanie: Cr [Ar] 3d4 4s2, Cu [Ar] 3d9 4s2\nstan podstawowy:       Cr [Ar] 3d5 4s1, Cu [Ar] 3d10 4s1")
        st.warning("Nie wyjaśniamy tych przypadków wyłącznie hasłem „półpełna lub pełna d jest stabilna”. Różnica wynika z bilansu całkowitej energii blisko leżących konfiguracji. Pełny atlas wyjątków należy do aplikacji 3.")
    st.subheader("Trzy zdania, które trzeba rozróżniać")
    st.markdown("1. **Zasada Aufbau:** stan podstawowy minimalizuje całkowitą energię.\n2. **Reguła n+l:** użyteczna heurystyka przewidywania kolejności zapełniania.\n3. **Dane referencyjne:** rzeczywiste konfiguracje i poziomy energii ustala spektroskopia oraz obliczenia.")
    q=st.radio("Które stwierdzenie jest poprawne?",["4s ma zawsze niższą energię niż 3d","3d ma zawsze niższą energię niż 4s","względna energia 4s i 3d zależy od atomu, obsadzenia i jonizacji"],index=None,key="quiz_4s")
    if q: st.success("Poprawnie." if q.startswith("względna") else "Nie istnieje jedna niezmienna kolejność dla wszystkich atomów i jonów.")

with t5:
    intro("Synteza i ćwiczenia","Połącz pojęcia: odpychanie elektronów zmienia odczuwane przyciąganie; penetracja różnicuje ekranowanie; energie podpowłok zależą od całego atomu.",["wyjaśniać łańcuch przyczynowy","rozwiązywać jakościowe problemy","przygotować się do konfiguracji atomów"])
    st.subheader("Mapa pojęć")
    st.markdown("**Ładunek jądra Z** + **rozmieszczenie elektronów** → **ekranowanie S** → **Z_eff**.\n\n**Kształt radialny orbitalu** → **penetracja** → inny średni kontakt z jądrem i ekranowaniem.\n\n**Z_eff + penetracja + odpychanie + korelacja + obsadzenie** → **energia całego stanu i względne energie podpowłok**.")
    st.subheader("Ćwiczenie 1")
    q1=st.radio("Przy tym samym n, który typ orbitalu ma zwykle największą penetrację?",["s","p","d"],index=None,key="ex1")
    if q1: st.success("Poprawnie." if q1=="s" else "Porównaj wewnętrzną część rozkładów radialnych.")
    st.subheader("Ćwiczenie 2")
    q2=st.radio("Czy większe Z_eff zwykle stabilizuje elektron?",["tak — silniejsze przyciąganie obniża energię","nie — zawsze podwyższa energię","Z_eff nie ma związku z energią"],index=None,key="ex2")
    if q2: st.success("Poprawnie, jakościowo." if q2.startswith("tak") else "Większe efektywne przyciąganie zwykle wiąże elektron silniej.")
    st.subheader("Ćwiczenie 3 — obliczenie Slatera")
    st.markdown("Wróć do zakładki 2, wybierz Na i elektron 3s. Zapisz wkłady do S, oblicz Z_eff i porównaj z wynikiem aplikacji.")

st.markdown("---")
st.header("Słownik i ograniczenia modelu")
cols=st.columns(3)
cols[0].info("**Ekranowanie S**\n\nPrzybliżona redukcja przyciągania jądrowego przez obecność innych elektronów.")
cols[1].info("**Z_eff=Z−S**\n\nEfektywny parametr przyciągania dla wskazanego elektronu.")
cols[2].info("**Penetracja**\n\nUdział gęstości orbitalu w obszarze blisko jądra.")