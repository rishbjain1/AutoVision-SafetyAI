 # 🚗 AutoVision — Vehicle Detection & Tesla Autopilot Safety Analysis

**Two-part AI/ML project** at the intersection of **autonomous vehicles (AV)** and **road safety analytics**:
- **Part 1 (Computer Vision):** Detect the **type of vehicle** and **localize** it with a **bounding box** from road images.
- **Part 2 (Data Science/NLP):** Analyze **Tesla Autopilot usage** and its relationship to **road safety outcomes** using the `Tesla - Deaths.csv` dataset.

---

## 📌 Business Scenario

Autonomous Vehicles (AV) and Intelligent Transport Systems (ITS) rely on **real-time vehicle detection** for tracking, counting, and incident response. In parallel, industry and regulators need **evidence-based analysis** of driver-assistance systems such as **Tesla’s Full Self-Driving (FSD) beta** (begun in Oct 2020, >100k users) to understand safety implications.

---

## 🧠 Objectives

### Part 1 — Object Detection
- Build an **AI model** (deep learning) that:
  - **Classifies** the **vehicle type** in an image, and
  - **Localizes** the vehicle via a **rectangular bounding box**.
- Run **inference** on sample images and evaluate detection accuracy.

**Dataset:** `Images.zip` — images of (autonomous) vehicles.

### Part 2 — Autopilot Safety Analytics
- Perform **preliminary inspection/cleaning** on `Tesla - Deaths.csv`.
- Conduct **EDA** on accident events across **date, year, state, country**.
- Analyze **victims, driver/occupant fatalities, cyclist/pedestrian involvement, collisions with other vehicles**, distribution across **models**, and **verified Autopilot deaths**.

**Dataset:** `Tesla - Deaths.csv` (rich schema with event meta, model, deaths, driver/occupant flags, AP claimed/verified, sources, notes).

---

## 🏗️ Part 1 — Computer Vision: Vehicle Type + Bounding Box

### Workflow
1. **Data setup**
   - Create parent project directory and child folders for **train/val/test**.
   - Unzip `Images.zip` (note: large archive — unzip time depends on compute).
2. **Preprocessing & Visualization**
   - Read images with **OpenCV**, standardize size, aspect ratio, and dtype.
   - (Optional) Plot class-balanced samples to sanity check labels.
3. **Modeling**
   - Build a **CNN-based detector** in **TensorFlow/Keras**.
   - **Transfer learning** backbone optional (e.g., MobileNetV2 / EfficientNet).
   - **Heads**:
     - **Classification head** → vehicle type (softmax).
     - **Regression head** → bounding box `(x_min, y_min, x_max, y_max)` (linear).
   - **Losses**:
     - Classification: `categorical_crossentropy`.
     - Bounding box: `Huber` / `Smooth L1` (or `MSE`) on normalized coords.
   - **Regularization**: `Dropout`, data augmentation.
   - **Callbacks**: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`.
4. **Training**
   - Train **without** augmentation to set a baseline.
   - Train **with** augmentation (flip/shift/brightness/hue) to improve generalization.
5. **Evaluation**
   - Metrics: classification **accuracy**, bbox **IoU**/localization error.
   - Plot **train/val curves** to check over/underfitting.
6. **Inference**
   - Run predictions on sample images and **draw bounding boxes + labels**.

### Tech Stack
`TensorFlow`, `Keras`, `OpenCV`, `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`

---

## 📊 Part 2 — Data Science: Tesla Autopilot Safety

### Dataset: `Tesla - Deaths.csv` (selected variables)
- **Case#**, **Date/Year**, **Country/State**, **Description**
- **Deaths**, **Tesla driver**, **Tesla occupant**
- **Other vehicle**, **Cyclists/Peds**, **TSLA+cycl/peds**
- **Model**, **Autopilot claimed**
- **Verified Tesla Autopilot Deaths**, **Verified + All Deaths Reported to NHTSA SGO**
- **Source**, **Note**, **(Deceased 1–4)**

### Workflow
1. **Preliminary Inspection & Cleaning**
   - Dtypes, missing values, duplicates.
   - Drop **non-analytic** or **PII-heavy** columns (e.g., raw link lists, individual names) and **redundant** fields.
2. **EDA — Event & Outcome Views**
   - **Temporal**: counts by **date**, **year**, and **weekday**.
   - **Geographic**: **state**, **country** distribution; map-ready aggregates.
   - **Severity**: distribution of **deaths per accident**; proportion where **driver** and/or **occupant** died.
   - **Vulnerable road users**: events with **cyclists/pedestrians** involved.
   - **Collisions**: frequency of **Tesla vs. other vehicles**.
   - **Model split**: event distribution across **Tesla models**.
   - **Autopilot lens**: distribution of **Verified Autopilot Deaths**.
3. **(Optional) NLP & Clustering on Descriptions**
   - Clean text (tokenize, lemmatize, POS tag) with **NLTK**.
   - Vectorize via **TF-IDF**; explore **KMeans**/**SVD** topic clusters for narrative patterns.

### Tech Stack
`Pandas`, `NumPy`, `Matplotlib`, `Seaborn`, `Missingno`, `Scikit-learn`, `NLTK` *(for optional narrative analysis)*

---

## ✅ Results (Summary)
- **Object Detection (Part 1):**
  - Trained a CNN that **classifies vehicle types** and **predicts bounding boxes**.
  - Demonstrated accurate **inference** on sample images with drawn boxes and labels.
- **Safety Analytics (Part 2):**
  - Produced **time-series and geo-level insights** on Tesla accident events.
  - Quantified **deaths per event**, **driver/occupant fatality shares**, **cyclist/pedestrian involvement**, and **collisions with other vehicles**.
  - Summarized **model-level distributions** and **verified Autopilot deaths**.

> This project showcases **end-to-end ML**: from **CV model training and inference** to **robust data analysis** (and optional **NLP clustering**) for decision support.

---





