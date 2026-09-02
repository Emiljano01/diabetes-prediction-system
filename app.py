
import sys, threading, warnings
warnings.filterwarnings("ignore")

def install_packages():
    import subprocess
    for pkg in ["numpy", "pandas", "scikit-learn"]:
        try: __import__(pkg if pkg != "scikit-learn" else "sklearn")
        except ImportError: subprocess.check_call([sys.executable,"-m","pip","install",pkg,"-q"])
install_packages()

import tkinter as tk
from tkinter import messagebox
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.impute import SimpleImputer


BG       = "#f0f4f8"
WHITE    = "#ffffff"
BLUE     = "#2563eb"
BLUE_LT  = "#eff6ff"
BLUE_DK  = "#1d4ed8"
GREEN    = "#16a34a"
GREEN_LT = "#f0fdf4"
RED      = "#dc2626"
RED_LT   = "#fef2f2"
YELLOW   = "#d97706"
YEL_LT   = "#fffbeb"
GRAY     = "#6b7280"
GRAY_LT  = "#f9fafb"
BORDER   = "#e5e7eb"
TEXT     = "#111827"
TEXT2    = "#6b7280"


def train_model():
    try:
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
                "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
        df = pd.read_csv(url, names=cols)
    except:
        np.random.seed(42); n=768; o=np.random.binomial(1,.35,n)
        df = pd.DataFrame({
            "Pregnancies":np.where(o,np.random.poisson(4,n),np.random.poisson(2,n)),
            "Glucose":np.where(o,np.random.normal(141,31,n),np.random.normal(110,26,n)).clip(44,199),
            "BloodPressure":np.random.normal(72,12,n).clip(24,122),
            "SkinThickness":np.random.normal(29,11,n).clip(7,63),
            "Insulin":np.where(o,np.random.normal(155,118,n),np.random.normal(68,98,n)).clip(14,846),
            "BMI":np.where(o,np.random.normal(35,7,n),np.random.normal(30,7,n)).clip(18,67),
            "DiabetesPedigreeFunction":np.random.exponential(.47,n).clip(.07,2.42),
            "Age":np.where(o,np.random.normal(37,11,n),np.random.normal(30,11,n)).clip(21,81),
            "Outcome":o})
    fcols = [c for c in df.columns if c!="Outcome"]
    for c in ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]:
        df[c] = df[c].replace(0,np.nan)
    X=df[fcols]; y=df["Outcome"]
    imp=SimpleImputer(strategy="median"); Xi=imp.fit_transform(X)
    sc=StandardScaler(); Xs=sc.fit_transform(Xi)
    Xt,Xv,yt,yv=train_test_split(Xs,y,test_size=.2,random_state=42,stratify=y)
    m=RandomForestClassifier(n_estimators=200,max_depth=8,class_weight="balanced",random_state=42)
    m.fit(Xt,yt)
    acc=accuracy_score(yv,m.predict(Xv))
    auc=roc_auc_score(yv,m.predict_proba(Xv)[:,1])
    return m,imp,sc,fcols,acc,auc,len(df)

FIELDS = [
    ("Pregnancies",             "Numri i Shtatzënive",        "herë",   0,  20, "0 nëse mashkull ose asnjëherë"),
    ("Glucose",                 "Glukoza pas OGTT (2h)",      "mg/dL", 44, 200, "Normal: 70–99  |  Pre-diabet: 100–125  |  Diabet: ≥126"),
    ("BloodPressure",           "Presioni Diastolik",         "mmHg",  24, 122, "Normal: 60–80  |  I lartë: ≥90"),
    ("SkinThickness",           "Trashësia Lëkurës Tricep",   "mm",     7,  99, "Masë dhjami nënlëkuror. Normal: 10–35"),
    ("Insulin",                 "Insulina Serike (2h)",       "μU/mL",  0, 846, "Lër 0 nëse nuk e di. Normal: 16–166"),
    ("BMI",                     "Indeksi Masës Trupore",      "kg/m²", 10,  70, "Nënpeshë <18.5 | Normal 18.5–24.9 | Mbipeshë 25–29.9 | Obez ≥30"),
    ("DiabetesPedigreeFunction","Historia Familjare",         "",    0.05,2.5, "0.07–0.4 e ulët | 0.4–0.8 mesatare | >0.8 e lartë"),
    ("Age",                     "Mosha",                      "vjeç",   1, 120, "Rreziku rritet ndjeshëm pas moshës 45"),
]

def bmi_category(v):
    if v<18.5: return "Nënpeshë","#f59e0b"
    if v<25:   return "Normal","#16a34a"
    if v<30:   return "Mbipeshë","#d97706"
    return "Obezitet","#dc2626"

def glucose_category(v):
    if v<100:  return "Normal","#16a34a"
    if v<126:  return "Pre-diabet","#d97706"
    return "Diabet (nivel)","#dc2626"

def bp_category(v):
    if v<80:   return "Normal","#16a34a"
    if v<90:   return "Kufitar","#d97706"
    return "I lartë","#dc2626"

def age_risk(v):
    if v<35:   return "Rrezik i ulët","#16a34a"
    if v<45:   return "Rrezik mesatar","#d97706"
    return "Rrezik i lartë","#dc2626"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sistemi i Klasifikimit te Diabetit")
        self.geometry("980x780"); self.minsize(900,700)
        self.configure(bg=BG)
        self.model=self.imp=self.sc=self.fcols=None
        self.entries={}
        self._ui(); self._train()

    def _ui(self):
        # Header
        hdr=tk.Frame(self,bg=BLUE,pady=16); hdr.pack(fill="x")
        tk.Label(hdr,text="🩺  Sistemi i Klasifikimit të Diabetit",
                 font=("Segoe UI",17,"bold"),bg=BLUE,fg="white").pack()
        tk.Label(hdr,text="Machine Learning  ·  Random Forest  ·  Analiza Klinike",
                 font=("Segoe UI",9),bg=BLUE,fg="#bfdbfe").pack()

        self.sv=tk.StringVar(value="⏳  Duke trajnuar modelin ML, ju lutem prisni...")
        sb=tk.Label(self,textvariable=self.sv,font=("Segoe UI",9),
                    bg=YEL_LT,fg=YELLOW,anchor="w",padx=14,pady=7,bd=0)
        sb.pack(fill="x"); self._sb=sb

        # Scroll
        outer=tk.Frame(self,bg=BG); outer.pack(fill="both",expand=True)
        cv=tk.Canvas(outer,bg=BG,highlightthickness=0)
        sb2=tk.Scrollbar(outer,orient="vertical",command=cv.yview)
        self.sf=tk.Frame(cv,bg=BG)
        self.sf.bind("<Configure>",lambda e:cv.configure(scrollregion=cv.bbox("all")))
        cv.create_window((0,0),window=self.sf,anchor="nw")
        cv.configure(yscrollcommand=sb2.set)
        cv.bind_all("<MouseWheel>",lambda e:cv.yview_scroll(-1*(e.delta//120),"units"))
        sb2.pack(side="right",fill="y"); cv.pack(side="left",fill="both",expand=True)

        self._form(); self._result_placeholder()

    def _card(self,parent,pady=(8,4)):
        f=tk.Frame(parent,bg=WHITE,relief="flat",bd=0)
        f.pack(fill="x",padx=18,pady=pady)
        tk.Frame(f,bg=BORDER,height=1).pack(fill="x",side="bottom")
        return f

    def _form(self):
        c=self._card(self.sf,(12,4))
        tk.Label(c,text="📋  Të Dhënat Klinike të Pacientit",
                 font=("Segoe UI",11,"bold"),bg=WHITE,fg=TEXT,
                 anchor="w",padx=14,pady=10).pack(fill="x")

        g=tk.Frame(c,bg=WHITE,padx=14,pady=6); g.pack(fill="x")
        for i,(field,label,unit,mn,mx,hint) in enumerate(FIELDS):
            r,col=divmod(i,2); col*=2
            wrap=tk.Frame(g,bg=WHITE); wrap.grid(row=r,column=col,sticky="nsew",padx=6,pady=4)

            top=tk.Frame(wrap,bg=WHITE); top.pack(fill="x")
            lbl=f"{label}"+(f"  [{unit}]" if unit else "")
            tk.Label(top,text=lbl,font=("Segoe UI",9,"bold"),bg=WHITE,fg=TEXT).pack(side="left")
            tk.Label(top,text=f"  {mn}–{mx}",font=("Segoe UI",8),bg=WHITE,fg=GRAY).pack(side="left")

            tk.Label(wrap,text=hint,font=("Segoe UI",8),bg=WHITE,fg=TEXT2,anchor="w").pack(fill="x")

            ef=tk.Frame(wrap,bg=BORDER,padx=1,pady=1); ef.pack(fill="x",pady=(2,0))
            e=tk.Entry(ef,font=("Segoe UI",11),bg=GRAY_LT,fg=TEXT,
                       insertbackground=BLUE,relief="flat",bd=6)
            e.pack(fill="x")
            e.bind("<FocusIn>", lambda ev,f=ef:f.configure(bg=BLUE))
            e.bind("<FocusOut>",lambda ev,f=ef:f.configure(bg=BORDER))
            self.entries[field]=e

        for col in [0,2]: g.columnconfigure(col,weight=1)

        # Buttons
        bf=tk.Frame(c,bg=WHITE,padx=14,pady=12); bf.pack(fill="x")
        self.abtn=tk.Button(bf,text="🔍  Analizo Pacientin",font=("Segoe UI",11,"bold"),
            bg=BLUE,fg="white",activebackground=BLUE_DK,activeforeground="white",
            relief="flat",padx=22,pady=9,cursor="hand2",
            command=self._analyze,state="disabled")
        self.abtn.pack(side="left")
        tk.Button(bf,text="🗑  Pastro",font=("Segoe UI",10),
            bg=GRAY_LT,fg=GRAY,relief="flat",padx=16,pady=9,cursor="hand2",
            command=self._clear).pack(side="left",padx=8)
        tk.Button(bf,text="Demo Diabetik",font=("Segoe UI",9),
            bg=RED_LT,fg=RED,relief="flat",padx=12,pady=9,cursor="hand2",
            command=lambda:self._demo("d")).pack(side="right",padx=4)
        tk.Button(bf,text="Demo Normal",font=("Segoe UI",9),
            bg=GREEN_LT,fg=GREEN,relief="flat",padx=12,pady=9,cursor="hand2",
            command=lambda:self._demo("n")).pack(side="right")

    def _result_placeholder(self):
        self.ph=tk.Frame(self.sf,bg=WHITE); self.ph.pack(fill="x",padx=18,pady=(4,16))
        tk.Label(self.ph,text="Rezultati do të shfaqet këtu pasi të klikoni 'Analizo Pacientin'.",
                 font=("Segoe UI",10),bg=WHITE,fg=TEXT2,pady=22).pack()
        self.rc=None

    def _train(self):
        def go():
            m,imp,sc,fc,acc,auc,n=train_model()
            self.model,self.imp,self.sc,self.fcols=m,imp,sc,fc
            self.after(0,lambda:self._trained(acc,auc,n))
        threading.Thread(target=go,daemon=True).start()

    def _trained(self,acc,auc,n):
        self.sv.set(f"✅  Model gati  ·  {n} pacientë trajnues  ·  Saktësi: {acc*100:.1f}%  ·  AUC-ROC: {auc:.3f}")
        self._sb.configure(bg=GREEN_LT,fg=GREEN)
        self.abtn.configure(state="normal")

    def _analyze(self):
        if not self.model: return
        vals={}
        for field,label,unit,mn,mx,hint in FIELDS:
            raw=self.entries[field].get().strip().replace(",",".")
            if raw=="" or raw=="0":
                if field=="Insulin": vals[field]=np.nan; continue
                elif raw=="":
                    messagebox.showerror("Gabim",f"'{label}' nuk mund të jetë bosh!"); return
            try: v=float(raw)
            except: messagebox.showerror("Gabim",f"'{label}': duhet të jetë numër!"); return
            if not(mn<=v<=mx):
                messagebox.showerror("Gabim",f"'{label}': vlera {v} jashtë ({mn}–{mx})!"); return
            vals[field]=v

        X=np.array([[vals[f] for f in self.fcols]])
        Xi=self.imp.transform(X); Xs=self.sc.transform(Xi)
        pred=self.model.predict(Xs)[0]
        prob=self.model.predict_proba(Xs)[0]
        self._show(pred,prob,vals,Xi[0])

    def _show(self,pred,prob,vals,iv):
       
        if self.rc: self.rc.destroy()
        self.ph.pack_forget()

        pd_=prob[1]*100; pn=prob[0]*100
        diab=(pred==1)
        mc=RED if diab else GREEN
        ml=RED_LT if diab else GREEN_LT

        self.rc=tk.Frame(self.sf,bg=WHITE); self.rc.pack(fill="x",padx=18,pady=(4,18))

        
        top=tk.Frame(self.rc,bg=ml,padx=20,pady=16); top.pack(fill="x")
        icon="⚠️  RREZIK I LARTË PËR DIABET" if diab else "✅  RREZIK I ULËT — GJENDJE NORMALE"
        tk.Label(top,text=icon,font=("Segoe UI",14,"bold"),bg=ml,fg=mc).pack(anchor="w")
        sub=("Modeli ML ka identifikuar shenja klinike të konsiderueshme që sugjerojnë diabet."
             if diab else
             "Parametrat klinikë janë brenda kufijve normalë. Nuk ka shenja alarmante.")
        tk.Label(top,text=sub,font=("Segoe UI",9),bg=ml,fg=TEXT2).pack(anchor="w",pady=(4,0))

       
        ps=tk.Frame(self.rc,bg=WHITE,padx=20,pady=12); ps.pack(fill="x")
        tk.Label(ps,text="Probabiliteti i Parashikimit",
                 font=("Segoe UI",10,"bold"),bg=WHITE,fg=TEXT).pack(anchor="w",pady=(0,8))
        for lbl,pct,col in [("Jo-Diabetik",pn,GREEN),("Diabetik",pd_,RED)]:
            row=tk.Frame(ps,bg=WHITE); row.pack(fill="x",pady=3)
            tk.Label(row,text=f"{lbl}",font=("Segoe UI",9),bg=WHITE,
                     fg=TEXT,width=13,anchor="w").pack(side="left")
            bg2=tk.Frame(row,bg=BORDER,height=20,width=380); bg2.pack(side="left",padx=6)
            bg2.pack_propagate(False)
            w=max(3,int(pct/100*380))
            tk.Frame(bg2,bg=col,width=w,height=20).place(x=0,y=0)
            tk.Label(row,text=f"{pct:.1f}%",font=("Segoe UI",10,"bold"),
                     bg=WHITE,fg=col).pack(side="left")

       
        mp=max(pd_,pn)
        ct,cc=(("Shumë e Lartë",GREEN) if mp>=85 else
               ("E Lartë",BLUE)        if mp>=70 else
               ("Mesatare",YELLOW)     if mp>=60 else ("E Ulët — Konsultohuni me mjek!",RED))
        cf=tk.Frame(ps,bg=WHITE); cf.pack(fill="x",pady=(4,0))
        tk.Label(cf,text="Besueshmëria:",font=("Segoe UI",9),bg=WHITE,fg=TEXT2).pack(side="left")
        tk.Label(cf,text=f"  {ct}",font=("Segoe UI",9,"bold"),bg=WHITE,fg=cc).pack(side="left")

        
        tk.Frame(self.rc,bg=BORDER,height=1).pack(fill="x",padx=20)
        an=tk.Frame(self.rc,bg=WHITE,padx=20,pady=12); an.pack(fill="x")
        tk.Label(an,text="🔬  Analiza Klinike e Parametrave",
                 font=("Segoe UI",10,"bold"),bg=WHITE,fg=TEXT).pack(anchor="w",pady=(0,8))

        fv={f:iv[i] for i,f in enumerate(self.fcols)}
        params=[
            ("Glukoza",       f"{fv['Glucose']:.0f} mg/dL",       *glucose_category(fv['Glucose'])),
            ("Presioni",      f"{fv['BloodPressure']:.0f} mmHg",  *bp_category(fv['BloodPressure'])),
            ("BMI",           f"{fv['BMI']:.1f} kg/m²",           *bmi_category(fv['BMI'])),
            ("Mosha",         f"{fv['Age']:.0f} vjeç",            *age_risk(fv['Age'])),
            ("Insulina",      f"{fv['Insulin']:.0f} μU/mL",
             ("E lartë" if fv['Insulin']>166 else "Normale"),
             (RED if fv['Insulin']>166 else GREEN)),
            ("Pedigree",      f"{fv['DiabetesPedigreeFunction']:.2f}",
             ("E lartë" if fv['DiabetesPedigreeFunction']>0.8 else
              "Mesatare" if fv['DiabetesPedigreeFunction']>0.4 else "E ulët"),
             (RED if fv['DiabetesPedigreeFunction']>0.8 else
              YELLOW if fv['DiabetesPedigreeFunction']>0.4 else GREEN)),
        ]
        grid=tk.Frame(an,bg=WHITE); grid.pack(fill="x")
        for i,(name,value,status,col) in enumerate(params):
            r,c2=divmod(i,3); c2*=1
            cell=tk.Frame(grid,bg=GRAY_LT,padx=12,pady=8,relief="flat")
            cell.grid(row=r,column=c2,sticky="nsew",padx=4,pady=4)
            tk.Label(cell,text=name,font=("Segoe UI",8),bg=GRAY_LT,fg=TEXT2).pack(anchor="w")
            tk.Label(cell,text=value,font=("Segoe UI",11,"bold"),bg=GRAY_LT,fg=TEXT).pack(anchor="w")
            sf2=tk.Frame(cell,bg=col,padx=6,pady=2); sf2.pack(anchor="w",pady=(2,0))
            tk.Label(sf2,text=status,font=("Segoe UI",8,"bold"),bg=col,fg="white").pack()
        for c3 in range(3): grid.columnconfigure(c3,weight=1)

       
        tk.Frame(self.rc,bg=BORDER,height=1).pack(fill="x",padx=20)
        rskf=tk.Frame(self.rc,bg=WHITE,padx=20,pady=12); rskf.pack(fill="x")
        tk.Label(rskf,text="⚡  Faktorët Kryesorë të Rrezikut",
                 font=("Segoe UI",10,"bold"),bg=WHITE,fg=TEXT).pack(anchor="w",pady=(0,6))

        risks=[]
        if fv["Glucose"]>=126:   risks.append(("Glukoza kritike",f"{fv['Glucose']:.0f} mg/dL — nivel diabetik (≥126 mg/dL)","Kryesor",RED))
        elif fv["Glucose"]>=100: risks.append(("Glukoza e ngritur",f"{fv['Glucose']:.0f} mg/dL — pre-diabet (100–125 mg/dL)","Mesatar",YELLOW))
        if fv["BMI"]>=30:        risks.append(("Obezitet",f"BMI {fv['BMI']:.1f} — rrit ndjeshëm rrezikun","Kryesor",RED))
        elif fv["BMI"]>=25:      risks.append(("Mbipeshë",f"BMI {fv['BMI']:.1f} — faktor rreziku","Mesatar",YELLOW))
        if fv["BloodPressure"]>=90: risks.append(("Presion i lartë",f"{fv['BloodPressure']:.0f} mmHg — hipertension","Rrezik",RED))
        if fv["Age"]>=45:        risks.append(("Moshë e rrezikshme",f"{fv['Age']:.0f} vjeç — rreziku dyfishohet pas 45","Kryesor",RED))
        if fv["DiabetesPedigreeFunction"]>=0.8:
            risks.append(("Histori familjare",f"Pedigree {fv['DiabetesPedigreeFunction']:.2f} — rrezik i lartë gjenetik","Gjenetik",RED))
        if fv["Insulin"]>200:    risks.append(("Insulinë shumë e lartë",f"{fv['Insulin']:.0f} μU/mL — rezistencë insulinike","Klinik",YELLOW))

        if risks:
            for name,desc,tag,col in risks:
                row2=tk.Frame(rskf,bg=WHITE); row2.pack(fill="x",pady=2)
                tf=tk.Frame(row2,bg=col,padx=5,pady=2); tf.pack(side="left")
                tk.Label(tf,text=tag,font=("Segoe UI",7,"bold"),bg=col,fg="white").pack()
                tk.Label(row2,text=f"  {name}:",font=("Segoe UI",9,"bold"),
                         bg=WHITE,fg=TEXT).pack(side="left")
                tk.Label(row2,text=f" {desc}",font=("Segoe UI",9),
                         bg=WHITE,fg=TEXT2).pack(side="left")
        else:
            tk.Label(rskf,text="✓  Asnjë faktor i dukshëm rreziku nuk u identifikua.",
                     font=("Segoe UI",9),bg=WHITE,fg=GREEN).pack(anchor="w")

        
        tk.Frame(self.rc,bg=BORDER,height=1).pack(fill="x",padx=20)
        recf=tk.Frame(self.rc,bg=WHITE,padx=20,pady=12); recf.pack(fill="x")
        tk.Label(recf,text="⚕️  Rekomandimet Mjekësore",
                 font=("Segoe UI",10,"bold"),bg=WHITE,fg=TEXT).pack(anchor="w",pady=(0,6))
        recs=([
            ("Konsultë Urgjente","Vizitoni endokrinolog ose mjek të familjes sa më shpejt"),
            ("Testime Laboratori","HbA1c, glukoza e agjërimit dhe profili lipidik"),
            ("Ndryshim Diete","Kufizoni sheqerin, karbohidratet e rafinuara dhe ushqimet e përpunuara"),
            ("Aktivitet Fizik","Minimumi 150 minuta aktivitet aerobik në javë (30 min/ditë)"),
            ("Monitorim","Kontrolloni rregullisht glukozën, BMI dhe presionin e gjakut"),
        ] if diab else [
            ("Jetesë e Shëndetshme","Vazhdoni me dietën e ekuilibruar dhe aktivitetin fizik"),
            ("Kontrolle Vjetore","Bëni testin e glukozës çdo vit për parandalim"),
            ("Peshë e Shëndetshme","Mbani BMI brenda 18.5–24.9 kg/m²"),
            ("Shmangni Rreziqet","Kufizoni alkoolin, shmangni duhanin, flini 7–8 orë"),
        ])
        for title2,desc in recs:
            row3=tk.Frame(recf,bg=BLUE_LT,padx=12,pady=6); row3.pack(fill="x",pady=2)
            tk.Label(row3,text=f"→  {title2}:",font=("Segoe UI",9,"bold"),
                     bg=BLUE_LT,fg=BLUE).pack(side="left")
            tk.Label(row3,text=f"  {desc}",font=("Segoe UI",9),
                     bg=BLUE_LT,fg=TEXT).pack(side="left")

        
        tk.Label(self.rc,
                 text="⚠  Ky aplikacion është vetëm për qëllime edukative dhe kërkimore. NUK zëvendëson diagnozën mjekësore profesionale.",
                 font=("Segoe UI",8),bg=YEL_LT,fg=YELLOW,pady=8,padx=14,anchor="w").pack(fill="x")

    def _clear(self):
        for e in self.entries.values(): e.delete(0,tk.END)
        if self.rc: self.rc.destroy(); self.rc=None
        self.ph.pack(fill="x",padx=18,pady=(4,16))

    def _demo(self,tip):
        self._clear()
        v={"d":{"Pregnancies":"6","Glucose":"168","BloodPressure":"92",
                "SkinThickness":"35","Insulin":"210","BMI":"38.5",
                "DiabetesPedigreeFunction":"1.2","Age":"52"},
           "n":{"Pregnancies":"1","Glucose":"92","BloodPressure":"68",
                "SkinThickness":"18","Insulin":"0","BMI":"22.1",
                "DiabetesPedigreeFunction":"0.18","Age":"27"}}[tip]
        for f,val in v.items():
            self.entries[f].delete(0,tk.END); self.entries[f].insert(0,val)

if __name__=="__main__":
    App().mainloop()