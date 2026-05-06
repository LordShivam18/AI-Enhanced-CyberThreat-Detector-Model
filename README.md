🛡️ CyberThreat-Detector-Model
This repository contains the core AI engine and real-time processing scripts for the AI-Enhanced Cybersecurity Threat Detector. The model is specifically trained to identify and classify sophisticated network attacks through probability analysis and frequency monitoring.

🚀 Overview
The system is designed to detect high-impact cybersecurity threats, including:

DDoS (Distributed Denial of Service)

Man-in-the-Middle (MitM)

SQL Injections

It leverages a machine learning model trained on 7,200 case files, providing a robust foundation for identifying anomalous network behavior.

📂 Project Structure
Based on the current repository state:

train_model.py: The core script used to train the model on the security datasets.

main.py: The entry point for the backend model and API serving.

kafka_producer.py: Utility to simulate network traffic by sending data packets to a Kafka topic.

kafka_consumer_test.py: A testing script to verify the ingestion and processing of data from the Kafka stream.

Dockerfile: Used for containerizing the model environment for consistent deployment.

requirements.txt: List of dependencies including PyTorch/TensorFlow, Kafka-Python, and Scikit-learn.

🛠️ Installation & Setup
Clone the repository:

Bash
git clone https://github.com/LordShivam18/CyberThreat-Detector-Model.git
cd CyberThreat-Detector-Model
Install dependencies:

Bash
pip install -r requirements.txt
Train the model (Optional):
If you wish to retrain or update the model with new data:

Bash
python train_model.py
📡 Real-Time Integration
This repo is built to work within a streaming architecture.

Use kafka_producer.py to push simulated network logs.

The main.py script (or Kafka consumer) analyzes the frequency and probability of the incoming logs to flag potential MitM or SQLi attempts.

📊 Probability Analysis
The model doesn't just provide a binary "Yes/No" for threats; it generates probability scores. This allows the production-grade system to:

Filter Noise: Ignore low-probability anomalies.

Generate Chart Tables: Visualize threat frequency over time in the frontend dashboard.

Predict Escalation: Identify when a series of minor events likely signals a coordinated DDoS attack.

🔗 Related Repositories
System Repo: [https://github.com/LordShivam18/CyberThreat-Detector] — This handles the React Dashboard and the primary PostgreSQL database for alert persistence.
