import os
import re
import nltk
import json
import faiss
import numpy as np
import pandas as pd
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer 

# ==== HF Token ====
HF_TOKEN = "Token_Here"
client = InferenceClient(token=HF_TOKEN)

# ==== NLTK setup ====
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ==== Load JSON data ====
json_path = r"HealthCareMagic-100k.json"
df = pd.read_json(json_path)
df = df.drop('instruction', axis=1)

# ==== Medical abbreviations ====
med_abbrev = {
    "abg": "arterial blood gases","ace": "angiotensin converting enzyme","acl": "anterior cruciate ligament","adhd": "attention deficit hyperactivity disorder","afib": "atrial fibrillation","aids": "acquired immunodeficiency syndrome",
    "alp": "alkaline phosphatase","als": "amyotrophic lateral sclerosis","alt": "alanine aminotransferase","amd": "age-related macular degeneration","ami": "acute myocardial infarction","aodm": "adult onset diabetes mellitus",
    "ast": "aspartate aminotransferase","avm": "arteriovenous malformation","bid": "twice a day","bmi": "body mass index","bp": "blood pressure","bph": "benign prostatic hypertrophy","brca": "breast cancer gene",
    "bun": "blood urea nitrogen","ca": "cancer","ca-125": "cancer antigen 125","cabg": "coronary artery bypass graft","cad": "coronary artery disease","cat": "computerized axial tomography","cbc": "complete blood count",
    "chd": "congenital heart disease","chf": "congestive heart failure","csf": "cerebrospinal fluid","cva": "cerebrovascular accident","cxr": "chest x-ray","d&c": "dilatation and curettage",
    "cmv": "cytomegalovirus","cns": "central nervous system","copd": "chronic obstructive pulmonary disease","cpk": "creatine phosphokinase","cpr": "cardiopulmonary resuscitation","crf": "chronic renal failure","crp": "c-reactive protein","djd": "degenerative joint disease","dm": "diabetes mellitus",
    "dtp": "diphtheria tetanus pertussis","dvt": "deep-vein thrombosis","dx": "diagnosis","ecg": "electrocardiogram","ekg": "electrocardiogram",
    "echo": "echocardiogram","eeg": "electroencephalogram",
    "emg": "electromyography","ent": "ear nose and throat",
    "ercp": "endoscopic retrograde cholangiopancreatography","esr": "erythrocyte sedimentation rate",
    "esrd": "end-stage renal disease","fsh": "follicle stimulating hormone",
    "gerd": "gastroesophageal reflux disease","gi": "gastrointestinal",
    "gfr": "glomerular filtration rate","gu": "genitourinary","hav": "hepatitis a virus","hbv": "hepatitis b virus",
    "hct": "hematocrit","hcv": "hepatitis c virus","hdl": "high density lipoprotein","hgb": "hemoglobin",
    "hiv": "human immunodeficiency virus","hpv": "human papilloma virus","hrt": "hormone replacement therapy",
    "htn": "hypertension","ibd": "inflammatory bowel disease","ibs": "irritable bowel syndrome","icd": "implantable cardioverter defibrillator",
    "icu": "intensive care unit","iddm": "insulin-dependent diabetes mellitus","im": "intramuscular",
    "iud": "intrauterine device","iv": "intravenous","ivp": "intravenous pyelogram","rsv": "respiratory syncytial virus",
    "rx": "prescription","sad": "seasonal affective disorder","sids": "sudden infant death syndrome","sle": "systemic lupus erythematosus",
    "sob": "shortness of breath","std": "sexually transmitted disease",
    "t3": "triiodothyronine","t4": "thyroxine","tb": "tuberculosis","tah": "total abdominal hysterectomy","tia": "transient ischemic attack",
    "tibc": "total iron binding capacity","tid": "three times a day","tmj": "temporomandibular joint",
    "torch": "group of infections tested for in newborns","tsh": "thyroid stimulating hormone","turp": "transurethral resection of prostate gland",
    "uri": "upper respiratory infection","uti": "urinary tract infection","xrt": "radiotherapy","wbc": "white blood cell"
}

# ==== Preprocessing functions ====
def md_links(text):
    return re.sub(r'\[.*?\]\(.*?\)', '', text)

def en_contractions(text):
    return ' '.join([contractions.fix(word) if word in contractions.contractions_dict else word
                     for word in text.split()])

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

def norm_lemma(text):
    tokens = text.split()
    clean_tokens = []
    for word in tokens:
        normalized = med_abbrev.get(word, word)
        for w in normalized.split():
            clean_tokens.append(lemmatizer.lemmatize(w))
    return ' '.join(clean_tokens)

# ==== Apply preprocessing ====
df['Patient_question'] = (df['input'].apply(preprocess_text)
                                      .apply(norm_lemma)
                                      .apply(md_links)
                                      .apply(en_contractions))

df['Doctor_answer'] = (df['output'].apply(preprocess_text)
                                   .apply(norm_lemma)
                                   .apply(md_links)
                                   .apply(en_contractions))

# ==== RAG setup ====
embedder = SentenceTransformer("all-MiniLM-L6-v2")

if os.path.exists("question_embeddings.npy"):
    question_embeddings = np.load("question_embeddings.npy")
else:
    question_embeddings = embedder.encode(df["Patient_question"].tolist(),
                                          convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(question_embeddings)
    np.save("question_embeddings.npy", question_embeddings)

# question_embeddings = embedder.encode(df["Patient_question"].tolist(),
#                                       convert_to_numpy=True, show_progress_bar=True)
faiss.normalize_L2(question_embeddings)
dim = question_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(question_embeddings)

# def get_similar_context(query, k=3):
#     q_embed = embedder.encode([query], convert_to_numpy=True)
#     faiss.normalize_L2(q_embed)
#     distances, indices = index.search(q_embed, k)
#     context_pairs = [(df.iloc[idx]["Patient_question"], df.iloc[idx]["Doctor_answer"]) for idx in indices[0]]
#     return context_pairs

def get_similar_context(query, k=3):
    q_embed = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_embed)
    distances, indices = index.search(q_embed, k)
    context_pairs = []
    for idx in indices[0]:
        if idx >= 0 and idx < len(df):
            context_pairs.append((df.iloc[idx]["Patient_question"], df.iloc[idx]["Doctor_answer"]))
    return context_pairs


# ==== Conversation history ====
conversation_history = []

def build_prompt(user_input, context_pairs):
    prompt = "Conversation:\n"
    for role, text in conversation_history[-10:]:
        prompt += f"{role}: {text}\n"
    prompt += f"Patient: {user_input}\n\nSimilar Q&A:\n"
    for i, (q, a) in enumerate(context_pairs):
        prompt += f"[{i+1}] Q: {q}\n[{i+1}] A: {a}\n"
    prompt += "\nDoctor:"
    return prompt[-1500:]

# ==== Mistral rewrite via HF API ====
def mistral_rewrite(question):
    prompt = f"Rewrite the following into a clear medical question:\n\n{question}\n\nRewritten:"
    messages = [{"role": "user", "content": prompt}]
    out = client.chat_completion(messages=messages, model="mistralai/Mistral-7B-Instruct-v0.2", max_tokens=100)
    text = out.choices[0].message.content
    return text.split("Rewritten:")[-1].strip()

# ==== Mistral conversational style ====
def mistral_conversational(text):
    prompt = f"Explain this medical answer in a warm, friendly doctor conversational style:\n\n{text}\n\nDoctor:"
    messages = [{"role": "user", "content": prompt}]
    out = client.chat_completion(messages=messages, model="mistralai/Mistral-7B-Instruct-v0.2", max_tokens=500)
    text = out.choices[0].message.content
    return text.split("Doctor:")[-1].strip()


# ==== Chat function ====
def chat_with_bot(user_input):
    conversation_history.append(("Patient", user_input))
    clarified = mistral_rewrite(user_input)
    context = get_similar_context(clarified)
    prompt = build_prompt(clarified, context)
    
    messages = [{"role": "user", "content": prompt}]
    out = client.chat_completion(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        messages=messages,
        max_tokens=150
    )
    bio_answer = out.choices[0].message.content.split("Doctor:")[-1].strip()
    
    final_answer = mistral_conversational(bio_answer)
    conversation_history.append(("Doctor", final_answer))
    return final_answer

def reset_chat():
    global conversation_history
    conversation_history = []

# ==== Main loop ====
if __name__ == "__main__":
    print("Type 'exit' to quit, 'reset' to clear chat history.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Conversation ended.")
            break
        elif user_input.lower() == "reset":
            reset_chat()
            continue
        answer = chat_with_bot(user_input)
        print("\nDoctor:", answer)


