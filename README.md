# A Comparative Study of Mouse, Trackpad, and Gesture Input in Fast Selection Tasks

This repository contains my HCI project comparing how the **mouse**, **trackpad**, and **gesture control** perform in fast clicking tasks.  
The aim was to measure speed, accuracy, workload, and overall usability across the three input modes.

The repo includes the experiment interface, participant datasets, analysis scripts, output graphs, and demo videos.


## 🎥 Demo Videos
Gesture interaction & demo recordings:  
https://drive.google.com/drive/folders/1hWs2Utl-AkeSdKYilQS1rfShfOew82CF?usp=sharing


## 📂 Folder Structure

```text
.
├── Dataset/          # Raw participant CSVs (trials, TLX, combined data)
├── outputs/          # Analysis results, graphs, summary tables
├── index.html        # The experiment (mouse / trackpad / gesture)
├── input_analysis.py # Data processing + visualization script
└── README.md
```


## 🌐 Live Website (Netlify)

[link](https://radiant-trifle-d09f70.netlify.app/)


## 🧪 About the Experiment

Participants completed rapid clicking tasks across different conditions:

- **Input Modes:** Mouse, Trackpad, Gesture  
- **Background:** Light / Dark  
- **Sound:** On / Off  

Each trial recorded:

- Time taken  
- Hit / Miss  
- Distance  
- Target size  
- Mode & condition  

After each block, participants completed a **NASA-TLX** workload form.


## 📊 Metrics Collected

- Movement time  
- Error rate  
- Fitts’ Law index of difficulty  
- Throughput  
- Learning curves  
- NASA-TLX workload  
- ANOVA + post-hoc statistical tests  

All summarized results are available in the `outputs/` folder.


## 🛠️ Tech Used

- **HTML / CSS / JavaScript**  
- **MediaPipe Hands** (gesture tracking)  
- **Python**  
  - NumPy  
  - Pandas  
  - Matplotlib  
  - SciPy  
  - Statsmodels  


## ✍️ About This Project

I built this project to understand how different input methods affect user performance in fast interaction tasks.  
It’s a mix of coding, UX thinking, and research — the type of work I enjoy doing and want to improve in.


## 🙌 Credits

Created by **Kishore S (2025)**  
