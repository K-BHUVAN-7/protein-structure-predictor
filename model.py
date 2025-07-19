import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, MultiHeadAttention, LayerNormalization, Dense, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gradio as gr
import matplotlib.pyplot as plt

# --- Constants ---
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWYX'
SEC_STRUCT_LABELS = 'HEC'
aa_to_int = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
sec_struct_to_int = {s: i for i, s in enumerate(SEC_STRUCT_LABELS)}
int_to_sec_struct = {i: s for s, i in sec_struct_to_int.items()}
vocab_size = len(AMINO_ACIDS) + 1
output_size = len(SEC_STRUCT_LABELS)
embedding_dim = 128
lstm_units = 64
max_sequence_length = 100
padding_label_value = output_size

# --- Dummy Training Data ---
sequences = ["ACDEFGHIKLMNPQRSTVWY", "MKTFFVAGL", "SEQUENCE"]
labels = ["HHHEECCCHHHHEECCCHH", "HHHHHHHHH", "CCCCCHHHH"]

def sequence_to_int(seq): return [aa_to_int.get(a, 0) for a in seq]
def labels_to_int(lbl): return [sec_struct_to_int.get(c, 0) for c in lbl]

X = [sequence_to_int(s) for s in sequences]
Y = [labels_to_int(l) for l in labels]

X_padded = pad_sequences(X, maxlen=max_sequence_length, padding='post', value=0)
Y_padded = pad_sequences(Y, maxlen=max_sequence_length, padding='post', value=padding_label_value)
sample_weight_mask = np.array([[1.0 if val != padding_label_value else 0.0 for val in row] for row in Y_padded], dtype=np.float32)

# --- Hybrid Model ---
input_seq = Input(shape=(max_sequence_length,))
embed = Embedding(vocab_size, embedding_dim, mask_zero=True)(input_seq)
lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(embed)
attn_out = MultiHeadAttention(num_heads=4, key_dim=embedding_dim)(lstm_out, lstm_out)
attn_norm = LayerNormalization()(attn_out + lstm_out)
output_logits = TimeDistributed(Dense(output_size, activation='softmax'))(attn_norm)

model = Model(inputs=input_seq, outputs=output_logits)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(ignore_class=padding_label_value),
              metrics=['accuracy'])

model.fit(X_padded, Y_padded, epochs=5, sample_weight=sample_weight_mask, validation_split=0.2)

# --- Prediction ---
def predict_structure(seq):
    clean_seq = seq.upper()
    ints = sequence_to_int(clean_seq)
    padded = pad_sequences([ints], maxlen=max_sequence_length, padding='post', value=0)
    preds = model.predict(padded)
    pred_idx = np.argmax(preds[0], axis=-1)[:len(clean_seq)]
    return ''.join([int_to_sec_struct.get(i, '?') for i in pred_idx])

# --- Visualizer ---
def visualize_prediction(sequence, predicted_labels):
    fig, ax = plt.subplots(figsize=(len(sequence)*0.4, 2.5))
    for i, aa in enumerate(sequence):
        ax.text(i, 2.0, aa, ha='center', va='center', fontsize=12, fontweight='bold')
    for i, label in enumerate(predicted_labels):
        color = {'H': 'red', 'E': 'blue', 'C': 'green'}.get(label, 'gray')
        ax.text(i, 1.0, label, ha='center', va='center', fontsize=12, color=color)
    ax.set_ylim(0.5, 2.5)
    ax.axis('off')
    plt.tight_layout()
    return fig

# --- Gradio Interface ---
def gradio_predict(seq):
    seq = seq.strip().upper()
    if not seq or not all(c in AMINO_ACIDS for c in seq):
        return "Invalid input. Use valid amino acids only.", None
    pred = predict_structure(seq)
    fig = visualize_prediction(seq, pred)
    return pred, fig

gradio_ui = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Textbox(label="Amino Acid Sequence"),
    outputs=[
        gr.Textbox(label="Predicted Secondary Structure (H/E/C)"),
        gr.Plot(label="Visualized Prediction")
    ],
    title="Protein Structure Predictor (Bi-LSTM + Transformer)",
    description="Enter amino acid sequence to predict secondary structure. Visualized per residue.",
)

print("🧬 Ready to launch! Run `gradio_ui.launch(share=True)` to test your model with visuals.")
