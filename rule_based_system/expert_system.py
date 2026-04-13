import pandas as pd
from experta import *
df = pd.read_csv('heart.csv')
df
df = df.dropna()
class Patient(Fact): 
    pass
class HeartDiseaseExpert(KnowledgeEngine):

    @Rule(Patient(age=P(lambda x: x > 50),
                  cholesterol=P(lambda x: x > 240)))
    def high_risk_age_chol(self):
        print("High Risk: Age > 50 and Cholesterol > 240")

    @Rule(Patient(bp=P(lambda x: x > 140),
                  smoking='yes'))
    def high_risk_bp_smoking(self):
        print("High Risk: High BP and Smoking")

    @Rule(Patient(diabetes='yes',
                  bmi=P(lambda x: x > 30)))
    def high_risk_diabetes_bmi(self):
        print("High Risk: Diabetes and Obesity")

    @Rule(Patient(chest_pain='typical',
                  max_hr=P(lambda x: x < 100)))
    def high_risk_chestpain_hr(self):
        print("High Risk: Typical chest pain + Low heart rate")

    @Rule(Patient(oldpeak=P(lambda x: x > 2)))
    def high_risk_oldpeak(self):
        print("High Risk: ST depression (oldpeak > 2)")



    @Rule(Patient(age=P(lambda x: 40 <= x <= 50),
                  cholesterol=P(lambda x: 200 <= x <= 240)))
    def medium_risk_age_chol(self):
        print("Medium Risk: Moderate age and cholesterol")

    @Rule(Patient(bp=P(lambda x: 120 <= x <= 140)))
    def medium_risk_bp(self):
        print("Medium Risk: Elevated BP")

    @Rule(Patient(bmi=P(lambda x: 25 <= x <= 30)))
    def medium_risk_bmi(self):
        print("Medium Risk: Overweight")


    @Rule(Patient(exercise='regular',
                  bmi=P(lambda x: x < 25)))
    def low_risk_exercise(self):
        print("Low Risk: Active lifestyle and healthy BMI")

    @Rule(Patient(smoking='no',
                  cholesterol=P(lambda x: x < 200)))
    def low_risk_smoking_chol(self):
        print("Low Risk: No smoking and good cholesterol")

    @Rule(Patient(age=P(lambda x: x < 40),
                  bp=P(lambda x: x < 120)))
    def low_risk_young_bp(self):
        print("Low Risk: Young with normal BP")

engine = HeartDiseaseExpert()
engine.reset
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
