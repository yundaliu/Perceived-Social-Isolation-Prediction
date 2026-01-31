import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login


def extractTokenEmbeddings(texts, args):
    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    else:
        raise EnvironmentError('Could not access GPU.')
        
    # Load a pretrained model and tokenizer
    # Can use other models like "roberta-base" or "distilbert-base-uncased"
    model_name = args['pretrained_model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Tokenize input text
    inputs = tokenizer(texts, 
                       padding=True, 
                       truncation=True, 
                       return_tensors='pt')
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract the last hidden state
    last_hidden_state = outputs.last_hidden_state
    
    # Extract embeddings
    # Shape: (batch_size, hidden_size)
    embeddings = last_hidden_state[:, 0, :]
    return embeddings.detach().cpu().numpy()


def extractSentenceEmbeddings(texts, args):
    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    else:
        raise EnvironmentError('Could not access GPU.')
        
    # Load a pretrained model and tokenizer
    # Can use other models like "roberta-base" or "distilbert-base-uncased"
    model_name = args['pretrained_model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    # Encode sentence
    inputs = tokenizer(texts,
                       padding=True, 
                       truncation=True, 
                       return_tensors='pt')
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Get token embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mean Pooling - Convert Token Embeddings to Sentence Embeddings
    token_embeddings = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
    attention_mask = inputs["attention_mask"]  # (batch_size, seq_len)

    # Apply Masked Mean Pooling
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
        input_mask_expanded.sum(dim=1), min=1e-9
    )
    return sentence_embeddings.detach().cpu().numpy()
    

def convert_text_to_embeddings(texts, args, batch_size=16):
    '''
    features: (n_samples, 1, n_texts)
    '''
    if texts.dtype == np.dtype(np.float64):
        raise ValueError('texts.dtype == np.dtype(np.float64).')
        
    if args['em_level'] == 'token':
        emFunc = extractTokenEmbeddings
    elif args['em_level'] == 'sentence':
        emFunc = extractSentenceEmbeddings
    
    # login('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')
    
    embeddings = []
    for idx in tqdm(range(0, len(texts), batch_size)):
        # define start_idx and end_idx
        start_idx, end_idx = idx, idx + batch_size

        # extract the batch
        text_arr = texts[start_idx:end_idx]
        n_samples = text_arr.shape[0]

        if args['text_level'] == 'question':
            # construct question list and id list
            question_list = []; id_list = []
            i = 0
            for questions in text_arr:
                for question in questions:
                    n_question = len(question)

                    # save
                    question_list.extend(question)
                    id_list.extend([i] * n_question)
                    i += 1
            id_list = np.array(id_list)

            # extract embeddings
            questionEm = emFunc(question_list, args)

            # save embeddings
            for uniq_id in np.unique(id_list):
                mask = id_list == uniq_id
                embeddings.append(questionEm[mask])
        elif args['text_level'] == 'survey':
            survey_list = []
            
            # concatenate questions to generate
            # a single paragraph
            for questions in text_arr:
                for question in questions:
                    survey_list.append(' '.join(question.tolist()))

            # extract embeddings
            surveyEm = emFunc(survey_list, args)

            # save embeddings
            embeddings.extend(surveyEm)
    return np.array(embeddings)


if __name__ == '__main__':
    # texts = [
    #     "Built environment barriers (e.g., no curb cuts, no accessbile ramp)",
    #     "Weather (e.g., temperature, humidity, rain, wind, air quality)",
    # ]
    # embeddings = extractTokenEmbeddings(texts)
    # print(embeddings.shape)
    
    texts = [
        "Built environment barriers (e.g., no curb cuts, no accessbile ramp)",
        "Weather (e.g., temperature, humidity, rain, wind, air quality)",
    ]
    
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    
    inputs = tokenizer(texts,
                       padding=True, 
                       truncation=True, 
                       return_tensors='pt')
    
    # Get token embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mean Pooling - Convert Token Embeddings to Sentence Embeddings
    token_embeddings = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
    attention_mask = inputs["attention_mask"]  # (batch_size, seq_len)

    # Apply Masked Mean Pooling
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
        input_mask_expanded.sum(dim=1), min=1e-9
    )
    print(sentence_embeddings.shape)