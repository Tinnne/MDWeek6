import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Map target values to species names
df['species'] = df['species'].map({i: species for i, species in enumerate(iris.target_names)})

# Streamlit app
def main():
    st.title("Iris Data Classification with Random Forest")
    
    # Sidebar settings
    st.sidebar.header("Model Hyperparameters")
    n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
    max_depth = st.sidebar.slider("Max Depth", 1, 10, 5)
    
    # Split data
    X = df[iris.feature_names]
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Progress bar
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)
        progress_bar.progress(percent_complete + 1)
    
    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"### Model Accuracy: {accuracy:.2f}")
    
    # Feature Importance
    feature_importance = pd.DataFrame({'Feature': iris.feature_names, 'Importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    
    st.write("### Feature Importance")
    fig, ax = plt.subplots()
    sns.barplot(x=feature_importance['Importance'], y=feature_importance['Feature'], ax=ax)
    st.pyplot(fig)
    
    # Scatter plot
    st.write("### Pairplot of Features")
    pairplot_fig = sns.pairplot(df, hue='species')
    st.pyplot(pairplot_fig)
    
    # User Input for Prediction
    st.write("### Predict Iris Species")
    sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.0)
    sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.0)
    petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, value=1.5)
    petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, value=0.2)
    
    if st.button("Predict"):
        input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = model.predict(input_data)
        predicted_species = iris.target_names[model.classes_.tolist().index(prediction[0])]
        st.write(f"### Predicted Species: {predicted_species}")

if __name__ == "__main__":
    main()