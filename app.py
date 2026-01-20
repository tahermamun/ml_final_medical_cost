import gradio as gr
import pickle
import pandas as pd



# load trained model
with open("./medical_insurance_model.pkl", "rb") as f:
    model = pickle.load(f)

# define a prediction function
def predict_insurance(age, sex, bmi, children, smoker, region):
    # Create dataframe for the input
    input_df = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })
    
    # predict using the loaded model
    pred = model.predict(input_df)[0]
    return round(pred, 2)

# gradio interface
app = gr.Interface(
    fn=predict_insurance,
    inputs=[
        gr.Number(label="Age"),
        gr.Dropdown(choices=["male", "female"], label="Sex"),
        gr.Number(label="BMI"),
        gr.Number(label="Number of Children"),
        gr.Dropdown(choices=["yes", "no"], label="Smoker"),
        gr.Dropdown(choices=["southwest", "southeast", "northwest", "northeast"], label="Region")
    ],
    outputs=gr.Number(label="Predicted Insurance Charges"),
    title="Medical Insurance Charges Predictor",
    description="Predict medical insurance charges based on personal details."
)

app.launch(share=True)
