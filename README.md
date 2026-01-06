# INFO 6000 – Informatics III (Graduate Level)
University of Georgia | Fall 2025

This repository contains selected coursework from **INFO 6000: Informatics III (Graduate Level)**, completed as part of my undergraduate and graduate-level studies at the University of Georgia.

The course emphasized advanced machine learning and deep learning techniques, reinforcement learning, large language models (LLMs), and industrial applications. Coursework included hands-on assignments, mini-projects, and a graduate-level final project using real-world datasets across domains such as healthcare, manufacturing, FinTech, and agriculture.

---

## Topics Covered
- Advanced data management, integration, and communication of structured and unstructured data
- Reinforcement learning concepts and applications
- Deep Learning: Transfer learning, CNNs, NLP, and Transformer-based models
- Deployment of predictive models using Streamlit, Flask, and SQL
- Special algorithms: Singular Value Decomposition (SVD) and others
- Audio analytics: preprocessing and feature extraction
- Large Language Models (LLMs) and application development
- Project-based workflows with real-world datasets

---

## Tools & Technologies
- Python, NumPy, Pandas, Matplotlib, Seaborn
- PyTorch, TensorFlow, scikit-learn
- Hugging Face Transformers & Trainer API
- OpenAI Gymnasium / Stable-Baselines3
- Streamlit & Flask for model deployment
- SQL, MQTT for data communication
- Jupyter Notebooks & VSCode

---

## Repository Structure
- `assignments/` – Programming exercises applying advanced ML/DL concepts
- `mini-projects/` – Applied projects demonstrating end-to-end workflows
- `final-project/` – Graduate-level Furuta Pendulum RL project

---

## Notable Work
- **Mini Project 1 – Medical Text Classification & Physics Q&A**  
Fine-tuned a BERT-based text classifier for medical diagnosis and deployed it via a Flask API and a Streamlit front-end. Built a physics question-answering system using Flan-T5 and retrieval-augmented generation, providing an interactive web interface for students to query and receive high-school-level explanations.

- **Mini Project 2 – Inventory Management & Valve Fault Detection**  
Developed a custom reinforcement learning environment for inventory management and trained a DQN agent to optimize stock levels, cash flow, and demand fulfillment. Extracted audio features from valve recordings and trained an SVM classifier to detect abnormal valve states, achieving 88–99% accuracy.

- **Mini Project 3 – Pump System Digital Twin**  
Created a Digital Twin of a pump system with a PI controller and variable frequency drive. Logged and visualized system metrics such as flow rate, RPM, power, and specific energy over time, performing validation scenarios and setpoint sweeps to analyze pump performance under varying conditions.

- **Final Project – Furuta Pendulum Reinforcement Learning**  
Implemented a full nonlinear Digital Twin of the Quanser QubeServo 2 (Furuta) pendulum, including RK4-based simulation, state normalization, reward computation, and optional real-time rendering. Trained a Soft Actor-Critic (SAC) agent with Stable-Baselines3 for reinforcement learning and evaluated the policy in real time, visualizing the rotary arm and pendulum while logging performance metrics.

---

## Notes
Datasets or trained models too large for GitHub are **omitted**, with instructions or references to access them.  

Repository reflects undergraduate and graduate-level coursework and has not been extensively refactored after submission.  

All work complies with the University of Georgia Academic Honesty Policy.  