# 🩺 Diabetes Prediction System

Sistem i klasifikimit të diabetit me Machine Learning (Random Forest), i ndërtuar me Python dhe Tkinter GUI. Analizon të dhëna klinike të pacientit dhe parashikon rrezikun e diabetit me analizë të detajuar dhe rekomandime mjekësore.

## ✨ Funksionalitete

- 🤖 Model Machine Learning (Random Forest Classifier) i trajnuar automatikisht
- 📋 Formular i plotë me 8 parametra klinikë (Glukoza, BMI, Presioni, etj.)
- 📊 Analizë e detajuar klinike me kategorizim të çdo parametri
- ⚡ Identifikim i faktorëve kryesorë të rrezikut
- ⚕️ Rekomandime mjekësore të personalizuara
- 🎨 GUI moderne dhe intuitive me Tkinter

## 🛠️ Teknologjitë e përdorura

- **Python 3.13**
- **Tkinter** – ndërfaqja grafike
- **scikit-learn** – modeli Machine Learning (Random Forest)
- **Pandas & NumPy** – përpunimi i të dhënave
- **Dataset:** Pima Indians Diabetes Dataset

## 🚀 Si të nisësh projektin lokalisht

1. Klono repository-n:
```bash
git clone https://github.com/Emiljano01/diabetes-prediction-system.git
cd diabetes-prediction-system
```

2. Krijo virtual environment dhe aktivizoje:
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux
```

3. Instalo librarite:
```bash
pip install -r requirements.txt
```

4. Nise aplikacionin:
```bash
python app.py
```