import pandas as pd
from experta import *
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

class Patient(Fact):
    pass

class HeartDiseaseExpert(KnowledgeEngine):

    @Rule(Patient(age=P(lambda x: x > 50), cholesterol=P(lambda x: x > 240)))
    def high_risk_age_chol(self):
        self.declare(Prediction(value=1))

    @Rule(Patient(bp=P(lambda x: x > 140), smoking='yes'))
    def high_risk_bp_smoking(self):
        self.declare(Prediction(value=1))

    @Rule(Patient(diabetes='yes', bmi=P(lambda x: x > 30)))
    def high_risk_diabetes_bmi(self):
        self.declare(Prediction(value=1))

    @Rule(Patient(chest_pain='typical', max_hr=P(lambda x: x < 100)))
    def high_risk_chestpain_hr(self):
        self.declare(Prediction(value=1))

    @Rule(Patient(oldpeak=P(lambda x: x > 2)))
    def high_risk_oldpeak(self):
        self.declare(Prediction(value=1))

    # Medium Risk
    @Rule(Patient(age=P(lambda x: 40 <= x <= 50), cholesterol=P(lambda x: 200 <= x <= 240)))
    def medium_risk(self):
        self.declare(Prediction(value=1))

    @Rule(Patient(bp=P(lambda x: 120 <= x <= 140)))
    def medium_risk_bp(self):
        self.declare(Prediction(value=1))

    @Rule(Patient(bmi=P(lambda x: 25 <= x <= 30)))
    def medium_risk_bmi(self):
        self.declare(Prediction(value=1))

    # Low Risk
    @Rule(Patient(exercise='regular', bmi=P(lambda x: x < 25)))
    def low_risk(self):
        self.declare(Prediction(value=0))

    @Rule(Patient(smoking='no', cholesterol=P(lambda x: x < 200)))
    def low_risk_smoking(self):
        self.declare(Prediction(value=0))

    @Rule(Patient(age=P(lambda x: x < 40), bp=P(lambda x: x < 120)))
    def low_risk_young(self):
        self.declare(Prediction(value=0))


class Prediction(Fact):
    pass


df = pd.read_csv('heart.csv')
df = df.dropna().reset_index(drop=True)
df = df.drop_duplicates().reset_index(drop=True)

engine = HeartDiseaseExpert()
y_pred = []


for i, row in df.iterrows():
    engine.reset()
    
    patient = Patient(
        age=int(row['age']),
        cholesterol=int(row['chol']),
        bp=int(row['trestbps']),
        smoking='yes' if row.get('smoking', 0) else 'no',
        bmi=float(row.get('bmi', 25)),
        exercise='regular' if row.get('exercise', 0) else 'no',
        diabetes='yes' if row.get('fbs', 0) else 'no',
        chest_pain='typical' if row['cp'] == 0 else 'atypical',
        max_hr=int(row['thalach']),
        oldpeak=float(row['oldpeak'])
    )
    
    engine.declare(patient)
    engine.run()
    
    pred = 0
    for fact in engine.facts.values():
        if isinstance(fact, Prediction):
            pred = fact['value']
            break
    
    y_pred.append(pred)

y_true = df['target'].values

print("\n=== تقييم Expert System ===")
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
def get_data():
    return Patient(
        age=int(input("Age: ")),
        cholesterol=int(input("Cholesterol: ")),
        bp=int(input("Blood Pressure: ")),
        smoking=input("Smoking (yes/no): "),
        bmi=float(input("BMI: ")),
        exercise=input("Exercise (yes/no): ")
    )

engine = HeartDiseaseExpert()
engine.reset()
engine.declare(get_data())
engine.run()