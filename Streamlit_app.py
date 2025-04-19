{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "a83e1405-eca5-4b68-acfe-df8f0c72f726",
   "metadata": {},
   "outputs": [
    {
     "ename": "FileNotFoundError",
     "evalue": "[Errno 2] No such file or directory: 'titanic_logistic_model.pkl'",
     "output_type": "error",
     "traceback": [
      "\u001b[31m---------------------------------------------------------------------------\u001b[39m",
      "\u001b[31mFileNotFoundError\u001b[39m                         Traceback (most recent call last)",
      "\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[1]\u001b[39m\u001b[32m, line 7\u001b[39m\n\u001b[32m      4\u001b[39m \u001b[38;5;28;01mimport\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01mjoblib\u001b[39;00m\n\u001b[32m      6\u001b[39m \u001b[38;5;66;03m# Load model and encoder\u001b[39;00m\n\u001b[32m----> \u001b[39m\u001b[32m7\u001b[39m model = \u001b[43mjoblib\u001b[49m\u001b[43m.\u001b[49m\u001b[43mload\u001b[49m\u001b[43m(\u001b[49m\u001b[33;43m'\u001b[39;49m\u001b[33;43mtitanic_logistic_model.pkl\u001b[39;49m\u001b[33;43m'\u001b[39;49m\u001b[43m)\u001b[49m\n\u001b[32m      8\u001b[39m sex_encoder = joblib.load(\u001b[33m'\u001b[39m\u001b[33msex_encoder.pkl\u001b[39m\u001b[33m'\u001b[39m)\n\u001b[32m     10\u001b[39m \u001b[38;5;66;03m# --- Streamlit Page Config ---\u001b[39;00m\n",
      "\u001b[36mFile \u001b[39m\u001b[32m~\\anaconda3\\Lib\\site-packages\\joblib\\numpy_pickle.py:650\u001b[39m, in \u001b[36mload\u001b[39m\u001b[34m(filename, mmap_mode)\u001b[39m\n\u001b[32m    648\u001b[39m         obj = _unpickle(fobj)\n\u001b[32m    649\u001b[39m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[32m--> \u001b[39m\u001b[32m650\u001b[39m     \u001b[38;5;28;01mwith\u001b[39;00m \u001b[38;5;28;43mopen\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43mfilename\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[33;43m'\u001b[39;49m\u001b[33;43mrb\u001b[39;49m\u001b[33;43m'\u001b[39;49m\u001b[43m)\u001b[49m \u001b[38;5;28;01mas\u001b[39;00m f:\n\u001b[32m    651\u001b[39m         \u001b[38;5;28;01mwith\u001b[39;00m _read_fileobject(f, filename, mmap_mode) \u001b[38;5;28;01mas\u001b[39;00m fobj:\n\u001b[32m    652\u001b[39m             \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(fobj, \u001b[38;5;28mstr\u001b[39m):\n\u001b[32m    653\u001b[39m                 \u001b[38;5;66;03m# if the returned file object is a string, this means we\u001b[39;00m\n\u001b[32m    654\u001b[39m                 \u001b[38;5;66;03m# try to load a pickle file generated with an version of\u001b[39;00m\n\u001b[32m    655\u001b[39m                 \u001b[38;5;66;03m# Joblib so we load it with joblib compatibility function.\u001b[39;00m\n",
      "\u001b[31mFileNotFoundError\u001b[39m: [Errno 2] No such file or directory: 'titanic_logistic_model.pkl'"
     ]
    }
   ],
   "source": [
    "# titanic_app_clean.py\n",
    "import streamlit as st\n",
    "import numpy as np\n",
    "import joblib\n",
    "\n",
    "# Load model and encoder\n",
    "model = joblib.load('titanic_logistic_model.pkl')\n",
    "sex_encoder = joblib.load('sex_encoder.pkl')\n",
    "\n",
    "# --- Streamlit Page Config ---\n",
    "st.set_page_config(page_title=\"Titanic Survival Predictor 🚢\", page_icon=\"🚢\", layout=\"centered\")\n",
    "\n",
    "# --- Main Title ---\n",
    "st.title(\"🚢 Titanic Survival Prediction App\")\n",
    "\n",
    "st.write(\n",
    "    \"\"\"\n",
    "    Enter passenger details in the sidebar to predict survival probability.\n",
    "    Model is trained on the classic Titanic dataset!\n",
    "    \"\"\"\n",
    ")\n",
    "\n",
    "# --- Sidebar for Inputs ---\n",
    "st.sidebar.header(\"Input Passenger Details\")\n",
    "\n",
    "pclass = st.sidebar.selectbox('Passenger Class', [1, 2, 3], format_func=lambda x: f\"{x} Class\")\n",
    "sex = st.sidebar.radio('Sex', ['male', 'female'])\n",
    "age = st.sidebar.slider('Age', 0, 100, 30)\n",
    "sibsp = st.sidebar.number_input('Number of Siblings/Spouses Aboard', min_value=0, max_value=10, value=0)\n",
    "parch = st.sidebar.number_input('Number of Parents/Children Aboard', min_value=0, max_value=10, value=0)\n",
    "fare = st.sidebar.slider('Fare Paid ($)', 0.0, 600.0, 32.0)\n",
    "\n",
    "# --- Prediction ---\n",
    "if st.sidebar.button('Predict Survival'):\n",
    "    with st.spinner('Predicting... ⏳'):\n",
    "        # Encode sex\n",
    "        sex_encoded = sex_encoder.transform([sex])[0]\n",
    "\n",
    "        # Prepare features\n",
    "        features = np.array([[pclass, sex_encoded, age, sibsp, parch, fare]])\n",
    "\n",
    "        # Make prediction\n",
    "        prediction = model.predict(features)\n",
    "        survival_prob = model.predict_proba(features)[0][1]\n",
    "\n",
    "        # --- Results ---\n",
    "        st.subheader(\"Prediction Results\")\n",
    "        if prediction[0] == 1:\n",
    "            st.success(f\"🎉 The passenger is likely to **SURVIVE**! (Probability: {survival_prob:.2f})\")\n",
    "        else:\n",
    "            st.error(f\"💀 Unfortunately, the passenger is likely to **NOT survive**. (Probability: {survival_prob:.2f})\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "97f2e986-aebd-4f9a-92e3-9de7f4dd38b3",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
