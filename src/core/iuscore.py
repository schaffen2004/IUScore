from sentence_transformers import SentenceTransformer
from src.utils.preprocess import get_information_units,normalize_iu
from src.core.extract_span import SpanExtracter
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import numpy as np
import torch

class IUScore:
    def __init__(self,
                 model_extraction="vinai/phobert-base",
                 model_encoding="all-mpnet-base-v2"
                 ):
        self.encoder = SentenceTransformer(model_encoding)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.span_extracter = SpanExtracter(model_name = model_extraction, device= self.device)
        self.log_path = None
    
    def similarity_matrix(self,gt_spans, ans_spans):

        emb_gt = self.encoder.encode(gt_spans)
        emb_ans = self.encoder.encode(ans_spans)

        return cosine_similarity(emb_gt, emb_ans)
    
    def compute_metrics(self,sim_matrix):


        sim_matrix = np.array(sim_matrix)

        num_gt, num_ans = sim_matrix.shape
        cost_matrix = -sim_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_scores = sim_matrix[row_ind, col_ind]

        precision = np.mean(matched_scores) if len(matched_scores) > 0 else 0
        recall = np.sum(matched_scores) / num_gt if num_gt > 0 else 0
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        uncertainties = []
        for j in range(num_ans):

            col = sim_matrix[:, j]

            best_idx = np.argmax(col)
            best_val = col[best_idx]

            if num_gt > 1:
                others = np.delete(col, best_idx)
                other_mean = np.mean(others)
            else:
                other_mean = 0

            gap = best_val - other_mean

            if best_val > 0:
                u = 1 - (gap / best_val)
            else:
                u = 1

            uncertainties.append(u)

        uncertainty = np.mean(uncertainties)

        return precision, recall, f1, uncertainty
    
    def __call__(self, question, gt_list, answer_list):
        
        # preprocessing
        answer_iu, gt_iu = get_information_units(answer_list, gt_list)
        
        print(answer_iu,gt_iu)
        
        # normalize
        answer_iu = normalize_iu(answer_iu)
        gt_iu = normalize_iu(gt_iu)
        
        print(answer_iu,gt_iu)
        
        # extract spans
        ans_spans = [self.span_extracter(question, iu) for iu in answer_iu]
        gt_spans = [self.span_extracter(question, iu) for iu in gt_iu]
        
        print(ans_spans,gt_spans)
        
        # similarity matrix
        sim_matrix = self.similarity_matrix(gt_spans, ans_spans)
        
        print(sim_matrix)
        
        # compute metrics
        precision, recall, f1, uncertainty = self.compute_metrics(sim_matrix)
        
        # result
        result = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "uncertainty": uncertainty
        }

        return result

if __name__ == "__main__":
    iu_score = IUScore()

    question = "Cá nhân đăng ký Bồi dưỡng nghiệp vụ đăng kiểm viên tàu cá phải nộp những loại giấy tờ gì?"
    gt = ['* Đơn đề nghị tham gia bồi dưỡng nghiệp vụ đăng kiểm viên tàu cá theo Mẫu số 01.ĐKV Phụ lục II ban hành kèm theo Thông tư này.']
    answer = ["* Cá nhân phải nộp bản sao văn bằng và chứng chỉ chuyên môn.  "]
    
    res = iu_score(question, gt, answer)
    print(res)