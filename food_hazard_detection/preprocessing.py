import pandas as pd
import json
import os
from nltk import download
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import re
import string
import contractions
download('averaged_perceptron_tagger')
download('stopwords')
download('punkt_tab')

def extract_json(response: str):
    """Extract JSON content from a formatted string."""
    match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = response.strip('```json').strip('```')

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Chyba při dekódování JSON: {e}")
        return None


def process_txt_files(folder_path, prefix):
    """Process all txt files starting with a prefix (e.g., 'food_hazar') in the folder and return a Pandas DataFrame."""
    all_data = []

    # Get files with prefix and end with .txt
    files = [f for f in os.listdir(folder_path) if f.startswith(prefix) and f.endswith(".txt")]

    # Order by number
    files.sort(key=lambda x: int(re.search(r'chunk(\d+)', x).group(1)))

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    json_obj = json.loads(line.strip())
                    content_str = json_obj.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message",
                                                                                                           {}).get(
                        "content", "")
                    extracted_json = extract_json(content_str)

                    if extracted_json:
                        row = {"id": json_obj["id"], "custom_id": json_obj["custom_id"]}
                        for feature in extracted_json.get("features", []):
                            row[feature["feature_name"]] = feature["answer"]

                        all_data.append(row)
                except json.JSONDecodeError:
                    print(f"Chyba dekódování JSON v souboru {filename}")

    df = pd.DataFrame(all_data)
    return df


def expand_contractions(text: str):
    """Expand contractions in the text."""
    return contractions.fix(text)


def split_hyphenated(tokens):
    """Split hyphenated words into individual words."""
    new_tokens = []
    for token in tokens:
        if "-" in token:
            new_tokens.extend(token.split("-"))  # Split into separate words
        else:
            new_tokens.append(token)
    return new_tokens


def remove_possessives(tokens):
    return [re.sub(r"'s\b", "", token) for token in tokens]


def get_wordnet_pos(word):
    """Map POS tag to first character for WordNetLemmatizer."""
    tag = pos_tag([word])[0][1][0].upper()
    return {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}.get(tag, wordnet.NOUN)


def tokenize(text: str):
    """Basic tokenization using regex and NLTK."""
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.replace('/', ' ')
    text = re.sub(r'\b(\w\.){2,}', lambda m: m.group(0).replace('.', ''), text)
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    return word_tokenize(text)


def remove_punctuation(tokens):
    """Remove punctuation from tokenized words."""
    return [word for word in tokens if word not in string.punctuation]


def remove_stopwords(tokens: list, stop_words: set):
    """Remove dynamically identified stopwords and non-alphabetic words."""
    return [word for word in tokens if word not in stop_words]


def lemmatize(tokens: list):
    """Lemmatize words using POS tags."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens]

def remove_special_characters(text: str):
    """
    Remove special characters from text while preserving spaces between words.
    """
    # Pattern to match special characters except spaces
    pattern = r'[^a-zA-Z0-9\s\-\']'
    return re.sub(pattern, '', text)

def preprocessing(text: str):
    """Full preprocessing pipeline with dynamic stopword removal."""
    text = expand_contractions(text)
    text = text.lower()
    text = remove_special_characters(text)
    tokens = tokenize(text)
    tokens = remove_possessives(tokens)
    tokens = remove_punctuation(tokens)
    stop_words = set(stopwords.words('english'))
    tokens = split_hyphenated(tokens)
    tokens = remove_stopwords(tokens, stop_words)
    tokens = lemmatize(tokens)
    return " ".join(tokens)
