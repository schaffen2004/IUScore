def get_information_units(answer,gt):
    # get information units from text
    answer_iu = [ iu for iu in answer if not (iu == '' or iu.endswith(":"))]
    gt_iu = [ iu for iu in gt if not (iu == '' or iu.endswith(":"))]
    return answer_iu, gt_iu

def normalize_iu(iu_list):
    cleaned = []

    for iu in iu_list:
        iu = iu.strip()                
        iu = iu.lstrip("*")            
        iu = iu.strip()                
        iu = iu.strip(" ;.*-")         

        cleaned.append(iu)

    return cleaned