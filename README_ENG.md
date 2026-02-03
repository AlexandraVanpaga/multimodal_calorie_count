# Multimodal AI for Food Calorie Estimation

A multimodal deep learning project for estimating the calorie content of dishes using text (ingredient lists) and image (food photos) inputs.

---

## Description

The project implements a multimodal model that predicts the calorie content of a dish by combining two input modalities:
- **Text** — ingredient list, processed by BERT
- **Image** — food photo, processed by EfficientNet-B7

The model fuses representations from both modalities to produce a single calorie prediction, achieving a validation MAE of ~40 kcal and R² ≈ 0.9.

### Key Features:
- Multimodal architecture: BERT (text) + EfficientNet-B7 (image)
- Selective layer unfreezing for fine-tuning pretrained encoders
- Image augmentation pipeline for improved generalization
- Ablation study to measure contribution of each modality
- Training visualization and loss monitoring
- No overfitting on the final model

---

## Project Structure

```
multimodal_calorie_count/
├── data/
│   ├── raw/                          # Raw data
│   └── processed/                    # Cleaned and joined data
├── src/
│   ├── get_raw_data.py               # Raw data download
│   ├── clean_data.py                 # Anomaly detection, data joining
│   ├── train.py                      # Model training
│   ├── ablation.py                   # Ablation study (modality importance)
│   └── train_viz.py                  # Training visualization
├── logs/
│   └── training_visualization.png    # Training plots
├── models/                           # Saved model checkpoints
├── requirements.txt
└── README.md
```

---

## Installation

### Requirements
- Python 3.8+
- PyTorch
- torchvision
- transformers (HuggingFace)
- timm

### Installing Dependencies

```bash
# Windows
.venv\Scripts\activate.bat

# Linux/Mac
source .venv/bin/activate

# Install packages
pip install -r requirements.txt
```

---

## Usage

### 1. Download Raw Data

```bash
python -m src.get_raw_data
```

### 2. Data Cleaning

Performs anomaly detection and joins across data sources.

```bash
python -m src.clean_data
```

### 3. Model Training

```bash
python -m src.train
```

### 4. Ablation Study

Evaluates the contribution of each modality (text-only, image-only, combined).

```bash
python -m src.ablation
```

### 5. Training Visualization

Generates loss and metric plots saved to `logs/training_visualization.png`.

```bash
python -m src.train_viz
```

---

## Model Architecture

```
Ingredient List (text) → BERT Encoder → Text Embedding
Food Photo (image)     → EfficientNet-B7 → Image Embedding
                                        ↓
                              Fusion Layer (concat)
                                        ↓
                              Regression Head → Calorie Prediction
```

### Fine-Tuning Strategy

Only select layers were unfrozen to prevent overfitting while adapting pretrained models to the food domain:

**BERT (text encoder):**
- Second-to-last and last transformer layers
- CLS token aggregation layer

**EfficientNet-B7 (image encoder):**
- Last convolutional block
- Final convolutional layer before classification
- Batch normalization layer

### Augmentation

- **Image augmentation:** Random crops, flips, color jitter, rotations — applied throughout training to improve generalization
- **Text augmentation:** Not used. Synonym substitution and back-translation produced incorrect or misleading ingredient names, hurting performance rather than helping

---

## 📈 Results

### Core Metrics

| Metric | Value |
|--------|-------|
| Validation MAE | ~40 kcal |
| R² | ~0.9 |
| Overfitting | None |

### Ablation Study Results

| Configuration | Relative Performance |
|---------------|---------------------|
| Text only (BERT) | Strong |
| Image only (EfficientNet-B7) | Moderate |
| **Text + Image (combined)** | **Best** |

The text modality proved slightly more important overall, as confirmed by the ablation study. However, the image modality provides a meaningful complementary signal — the combined model outperforms either branch alone.

### Model Comparison

| Visual Encoder | Result |
|----------------|--------|
| EfficientNet-B7 | Best |
| ViT | Worse (likely needs more data) |

---

## Error Analysis

The highest prediction errors were observed on two types of dishes:

- **Visually complex dishes** — where the image contains multiple ingredients or overlapping textures, making it harder for the CNN to extract relevant features
- **Dishes containing almonds** — the model likely confuses almonds with visually similar but lower-calorie ingredients. This can potentially be addressed by augmenting almond-heavy examples and retraining

---

## Conclusions

- The multimodal approach (BERT + EfficientNet-B7) delivers satisfactory calorie estimation from ingredient lists and food photos
- Selective layer unfreezing allowed effective fine-tuning without overfitting
- Text modality is slightly more important than image, but combining both yields the best results
- R² ≈ 0.9 indicates the model explains ~90% of calorie variation in the data
- Further improvements are possible through targeted augmentation of high-error examples (e.g., almond-containing dishes) and stronger image augmentation pipelines

---

## Technologies

- **PyTorch** — training framework
- **Transformers** — BERT text encoder (HuggingFace)
- **timm** — EfficientNet-B7 image encoder
- **torchvision** — image augmentation and preprocessing
- **pandas / numpy** — data processing
- **matplotlib / seaborn** — training visualization
EOF
Salida

# Multimodal AI for Food Calorie Estimation

A multimodal deep learning project for estimating the calorie content of dishes using text (ingredient lists) and image (food photos) inputs.

---

## Description

The project implements a multimodal model that predicts the calorie content of a dish by combining two input modalities:
- **Text** — ingredient list, processed by BERT
- **Image** — food photo, processed by EfficientNet-B7

The model fuses representations from both modalities to produce a single calorie prediction, achieving a validation MAE of ~40 kcal and R² ≈ 0.9.

### Key Features:
- Multimodal architecture: BERT (text) + EfficientNet-B7 (image)
- Selective layer unfreezing for fine-tuning pretrained encoders
- Image augmentation pipeline for improved generalization
- Ablation study to measure contribution of each modality
- Training visualization and loss monitoring
- No overfitting on the final model

---

## Project Structure

```
multimodal_calorie_count/
├── data/
│   ├── raw/                          # Raw data
│   └── processed/                    # Cleaned and joined data
├── src/
│   ├── get_raw_data.py               # Raw data download
│   ├── clean_data.py                 # Anomaly detection, data joining
│   ├── train.py                      # Model training
│   ├── ablation.py                   # Ablation study (modality importance)
│   └── train_viz.py                  # Training visualization
├── logs/
│   └── training_visualization.png    # Training plots
├── models/                           # Saved model checkpoints
├── requirements.txt
└── README.md
```

---

## Installation

### Requirements
- Python 3.8+
- PyTorch
- torchvision
- transformers (HuggingFace)
- timm

### Installing Dependencies

```bash
# Windows
.venv\Scripts\activate.bat

# Linux/Mac
source .venv/bin/activate

# Install packages
pip install -r requirements.txt
```

---

## Usage

### 1. Download Raw Data

```bash
python -m src.get_raw_data
```

### 2. Data Cleaning

Performs anomaly detection and joins across data sources.

```bash
python -m src.clean_data
```

### 3. Model Training

```bash
python -m src.train
```

### 4. Ablation Study

Evaluates the contribution of each modality (text-only, image-only, combined).

```bash
python -m src.ablation
```

### 5. Training Visualization

Generates loss and metric plots saved to `logs/training_visualization.png`.

```bash
python -m src.train_viz
```

---

## Model Architecture

```
Ingredient List (text) → BERT Encoder → Text Embedding
Food Photo (image)     → EfficientNet-B7 → Image Embedding
                                        ↓
                              Fusion Layer (concat)
                                        ↓
                              Regression Head → Calorie Prediction
```

### Fine-Tuning Strategy

Only select layers were unfrozen to prevent overfitting while adapting pretrained models to the food domain:

**BERT (text encoder):**
- Second-to-last and last transformer layers
- CLS token aggregation layer

**EfficientNet-B7 (image encoder):**
- Last convolutional block
- Final convolutional layer before classification
- Batch normalization layer

### Augmentation

- **Image augmentation:** Random crops, flips, color jitter, rotations — applied throughout training to improve generalization
- **Text augmentation:** Not used. Synonym substitution and back-translation produced incorrect or misleading ingredient names, hurting performance rather than helping

---

## 📈 Results

### Core Metrics

| Metric | Value |
|--------|-------|
| Validation MAE | ~40 kcal |
| R² | ~0.9 |
| Overfitting | None |

### Ablation Study Results

| Configuration | Relative Performance |
|---------------|---------------------|
| Text only (BERT) | Strong |
| Image only (EfficientNet-B7) | Moderate |
| **Text + Image (combined)** | **Best** |

The text modality proved slightly more important overall, as confirmed by the ablation study. However, the image modality provides a meaningful complementary signal — the combined model outperforms either branch alone.

### Model Comparison

| Visual Encoder | Result |
|----------------|--------|
| EfficientNet-B7 | Best |
| ViT | Worse (likely needs more data) |

---

## Error Analysis

The highest prediction errors were observed on two types of dishes:

- **Visually complex dishes** — where the image contains multiple ingredients or overlapping textures, making it harder for the CNN to extract relevant features
- **Dishes containing almonds** — the model likely confuses almonds with visually similar but lower-calorie ingredients. This can potentially be addressed by augmenting almond-heavy examples and retraining

---

## Conclusions

- The multimodal approach (BERT + EfficientNet-B7) delivers satisfactory calorie estimation from ingredient lists and food photos
- Selective layer unfreezing allowed effective fine-tuning without overfitting
- Text modality is slightly more important than image, but combining both yields the best results
- R² ≈ 0.9 indicates the model explains ~90% of calorie variation in the data
- Further improvements are possible through targeted augmentation of high-error examples (e.g., almond-containing dishes) and stronger image augmentation pipelines

---

## Technologies

- **PyTorch** — training framework
- **Transformers** — BERT text encoder (HuggingFace)
- **timm** — EfficientNet-B7 image encoder
- **torchvision** — image augmentation and preprocessing
- **pandas / numpy** — data processing
- **matplotlib / seaborn** — training visualization
