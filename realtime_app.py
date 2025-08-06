import streamlit as st
import torch
import numpy as np
import time

labels = ["Healthy", "Minor", "Severe"]

class DummyModel(torch.nn.Module):
    def forward(self, x_sensor, x_meta):
        probs = torch.softmax(torch.randn((1, 3)) * 3, dim=1)
        return probs

model = DummyModel()
model.eval()

st.set_page_config(page_title="🛠️ SHM Real-Time Monitor", layout="centered")
st.title("🛠️ Real-Time Bridge Health Monitor")
st.caption("Live streaming predictions using MetaConvLSTM")

placeholder = st.empty()
run = st.toggle("▶ Start Simulation", value=False)

if run:
    for i in range(50):
        x_sensor = np.random.randn(5, 10)
        x_meta = np.array([45, 12.5, 1, 0])

        x_sensor_tensor = torch.tensor(x_sensor, dtype=torch.float32).unsqueeze(0)
        x_meta_tensor = torch.tensor(x_meta, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            probs = model(x_sensor_tensor, x_meta_tensor).numpy()[0]
            pred = np.argmax(probs)

        with placeholder.container():
            st.markdown(f"### 🕒 Time Step {i+1}")
            st.metric("Prediction", labels[pred])
            st.progress(probs[pred])
            st.text(f"Confidence: {probs[pred]:.2f}")
            st.text(f"Raw Probabilities: {probs.round(3)}")
        time.sleep(1)
