# 🧬 Protein Structure Predictor (Bi-LSTM + Transformer)

This project is a hybrid deep learning model combining **Bi-directional LSTM** and **Transformer-based attention** to predict the **secondary structure of proteins** (Helix `H`, Strand `E`, Coil `C`) from amino acid sequences.

Built using:
- **TensorFlow & Keras**
- **Gradio** for interactive web UI
- **Matplotlib** for visualization

## 📌 Features

- Converts input amino acid sequences to secondary structure.
- Visualizes predictions with color-coded residues:
  - **Red**: Helix (H)
  - **Blue**: Strand (E)
  - **Green**: Coil (C)
- Easy-to-use web UI with Gradio.

## 🏗️ Model Architecture

- **Embedding Layer**
- **Bi-directional LSTM**
- **Multi-head Self Attention**
- **Layer Normalization**
- **TimeDistributed Dense Layer (Softmax)**

## 📦 Dependencies

Install dependencies via pip:

```bash
pip install tensorflow numpy matplotlib gradio
