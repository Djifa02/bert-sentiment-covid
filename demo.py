# =======================================================================================
# Interface de demonstration Gradio pour la classification de sentiment des tweets COVID
# =======================================================================================

import torch
import gradio as gr
from transformers import AutoTokenizer
from model import get_model
from dataset import MODEL_NAME, MAX_LENGTH, LABEL2ID, ID2LABEL

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH  = "best_model.pth"
NUM_CLASSES = 5

# ============================================================
# CHARGEMENT DU MODELE
# ============================================================
print("Chargement du modele...")

device    = torch.device("cuda" if torch.cuda.is_available()
                         else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = get_model(num_classes=NUM_CLASSES)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)
model.to(device)
model.eval()

print("Modele charge avec succes !")


# ============================================================
# FONCTION DE PREDICTION
# ============================================================
def predict_sentiment(text: str):

    if not text.strip():
        return "Veuillez saisir un tweet", {}

    # Tokenization
    encoding = tokenizer(
        text,
        max_length     = MAX_LENGTH,
        padding        = "max_length",
        truncation     = True,
        return_tensors = "pt"
    )

    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Prediction
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs  = torch.softmax(logits, dim=1)
        pred   = probs.argmax(dim=1).item()

    # Preparer les resultats
    label      = ID2LABEL[pred]
    probs_dict = {
        ID2LABEL[i]: float(probs[0][i])
        for i in range(NUM_CLASSES)
    }

    return label, probs_dict


# ============================================================
# INTERFACE GRADIO
# ============================================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # COVID-19 Tweet Sentiment Analyzer
    ### Analyse le sentiment des tweets sur le COVID-19

    Ce modele utilise BERT fine-tune sur des tweets COVID
    pour predire le sentiment parmi 5 classes :
    Extremely Positive - Positive - Neutral -
    Negative - Extremely Negative
    """)

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label       = "Tweet",
                placeholder = "Entrez un tweet sur le COVID-19...",
                lines       = 3
            )
            submit_btn = gr.Button(
                "Analyser le sentiment",
                variant = "primary"
            )

        with gr.Column():
            label_output = gr.Label(
                label = "Sentiment predit"
            )
            probs_output = gr.Label(
                label           = "Probabilites par classe",
                num_top_classes = 5
            )

    # Exemples pre-remplis
    gr.Examples(
        examples = [
            ["advice Talk to your neighbours family to exchange phone numbers create contact list with phone numbers of neighbours"],
            ["For corona prevention,we should stop to buy things with the cash and should use online payment methods"],
            ["@MeNyrbie @Phil_Gahan @Chrisitv https://t.co/iFz9FAn2Pa and https://t.co/xX6ghGFzCC"],
            ["Due to the Covid-19 situation, we have increased demand for all food products."],
            ["Me, ready to go at supermarket during the #COVID19 outbreak. Not because I'm paranoid"]
        ],
        inputs = text_input
    )

    gr.Markdown("""
    ---
    Auteurs : Adji Fatou NGOM et Viwossin DEGBOE
    Modele  : BERT fine-tune sur Corona NLP Dataset
    Classes : 5 sentiments
    """)

    submit_btn.click(
        fn      = predict_sentiment,
        inputs  = text_input,
        outputs = [label_output, probs_output]
    )


# ============================================================
# LANCEMENT
# ============================================================
if __name__ == "__main__":
    demo.launch(share=True)