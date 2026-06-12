The dataset used in this project is available upon request. Please feel free to contact me if you would like access to it.
# Automatic Detection of Patient-Ventilator Asynchronies


This project was developed as my Graduation Project (PFE) for the Bachelor's degree in Biomedical Instrumentation and Maintenance at the Institut Supérieur des Sciences de la Santé (ISSS), Settat.


## 📝 Project Overview
The main objective of this work is to provide a Clinical Decision Support System (CDSS) for intensive care. The model is designed to automatically identify and classify patient-ventilator asynchronies (PVAs) based on real-time analysis of airway pressure and flow waveforms.

- **Domain:** Biomedical Engineering & Artificial Intelligence.
- **Approach:** Deep Learning (1D-CNN).
- **Performance:** 96% accuracy in detecting premature cycling asynchronies.

## 🛠 Technical Architecture
The data pipeline and model architecture include:
1. **Data Pipeline:** Preprocessing and segmentation of over 63,857 respiratory cycles from simulated datasets.
2. **Architecture:** A 1D Convolutional Neural Network (CNN) implemented in PyTorch.
3. **Training:** Accelerated training using GPU (CUDA) support for high-frequency signal processing.

## 📁 Repository Structure
```bash
├── Project_Report               # Graduation Project Report (PDF) and Presentation (PPT)
├── main.py                 # Python source code (main.py, preprocessing scripts)
├── README.md             # Project documentation
└── .gitignore            # Git ignore file

