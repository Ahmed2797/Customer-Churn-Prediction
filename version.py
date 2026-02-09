from project.utils import load_object
from project.entity.estimator import ProjectModel
import pandas as pd

preprocessor = load_object("final_model/preprocessing.pkl")
model = load_object("final_model/best_model.pkl")
predict_pipeline = ProjectModel(transform_object=preprocessor, best_model_details=model)

test_df = pd.DataFrame([{
    "CreditScore": 600,
    "Age": 40,
    "Tenure": 3,
    "Balance": 50000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 50000,
    "Geography": "France",
    "Gender": "Male"
}])

print(preprocessor.feature_names_in_)
#print(preprocessor.transformers_)


print(type(test_df))
print(test_df)

print(model.predict(test_df))

