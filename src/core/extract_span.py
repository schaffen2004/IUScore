import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from captum.attr import IntegratedGradients

class SpanExtracter:
    def __init__(self,model_name = 'vinai/phobert-base',device=torch.device("cpu")):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        self.model = AutoModel.from_pretrained(model_name,output_attentions=True).to(self.device)
        self.model.eval()
    
    def encode_input(self, question, answer):
        text = question + " </s> " + answer

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        return input_ids, attention_mask, tokens
    
    @staticmethod
    def get_boundaries(tokens):

        sep_index = tokens.index("</s>")

        question_start = 1
        question_end = sep_index

        answer_start = sep_index + 1
        answer_end = len(tokens) - 1

        return question_start, question_end, answer_start, answer_end
    
    def cross_attention(self,attentions, q_start, q_end, a_start, a_end):

        att = torch.stack(attentions)

        att = att.mean(dim=0)
        att = att.mean(dim=1)

        att = att.squeeze()

        importance = torch.zeros(att.shape[0]).to(self.device)

        q_indices = list(range(q_start, q_end))
        a_indices = list(range(a_start, a_end))

        for j in a_indices:

            score = att[q_indices, j].mean()

            importance[j] = score

        return importance
    
    def gradient_importance(self,input_ids, attention_mask):

        embeddings = self.model.embeddings.word_embeddings(input_ids)

        embeddings = embeddings.clone().detach().requires_grad_(True)

        outputs = self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask
        )

        hidden = outputs.last_hidden_state

        score = hidden.mean()

        self.model.zero_grad()

        score.backward()

        grads = embeddings.grad

        grad_importance = torch.norm(grads, dim=-1)

        return grad_importance.squeeze()

    
    def integrated_gradients(self,input_ids, attention_mask):

        embeddings = self.model.embeddings.word_embeddings(input_ids)

        def forward_func(embeds):

            outputs = self.model(
                inputs_embeds=embeds,
                attention_mask=attention_mask
            )

            hidden = outputs.last_hidden_state
            return hidden.mean(dim=(1,2))

        ig = IntegratedGradients(forward_func)
        attributions = ig.attribute(embeddings)
        scores = torch.norm(attributions, dim=-1)

        return scores.squeeze()  
    
    @staticmethod
    def normalize(x):

        x = x.detach().cpu().numpy()

        return (x - x.min()) / (x.max() - x.min() + 1e-8)    
    
    def combine_scores(self,att, grad, ig):
        a = self.normalize(att)
        g = self.normalize(grad)
        i = self.normalize(ig)
        strength = np.array([a.var(), g.var(), i.var()]) + 1e-8  
        weights = strength / strength.sum()
        
        combined_weights = weights[0]*a + weights[1]*g + weights[2]*i
        
        return combined_weights
    
    @staticmethod
    def extract_best_span(scores, a_start, a_end, max_len=8):

        best_score = -1
        best_span = (a_start, a_start)

        for i in range(a_start, a_end):

            for j in range(i, min(i + max_len, a_end)):

                score = scores[i:j+1].sum()

                if score > best_score:

                    best_score = score
                    best_span = (i,j)

        return best_span    
    
    def __call__(self, question, answer):
        # encode question and answer
        input_ids, attention_mask, tokens = self.encode_input(question, answer)
        
        # find boudaries of question-answer
        q_start, q_end, a_start, a_end = self.get_boundaries(tokens)
        
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        attentions = outputs.attentions
        
        # compute cross attention 
        attention_scores = self.cross_attention(
            attentions,
            q_start,
            q_end,
            a_start,
            a_end
        )
        
        # gradient important
        grad_scores = self.gradient_importance(
            input_ids,
            attention_mask
        )
        
        # integrated gradients
        ig_scores = self.integrated_gradients(
            input_ids,
            attention_mask
        )
        
        # combine scores
        combined_scores = self.combine_scores(
            attention_scores,
            grad_scores,
            ig_scores
        )
        
        span = self.extract_best_span(
            combined_scores,
            a_start,
            a_end
        )
              
        best_tokens = tokens[span[0]:span[1]+1]
        
        return self.tokenizer.convert_tokens_to_string(best_tokens)
