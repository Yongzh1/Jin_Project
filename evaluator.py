from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, auc, f1_score

class Evaluator:
    def __init__(self, y_test, y_prob):
        self.y_test = y_test
        self.y_prob = y_prob


    def f1(self):
        y_pred = (self.y_prob >= 0.5).astype(int)  
        f1 = f1_score(self.y_test, y_pred)
        print(f'F1-score:{f1}')
        return f1

    def pr_auc(self):
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_prob)
        pr_auc = auc(recall,precision)
        print(f'PR-AUC:{pr_auc}')
        return pr_auc

    def roc_auc(self):
        roc_auc = roc_auc_score(self.y_test, self.y_prob)
        print(f'ROC-AUC:{roc_auc}')
        return roc_auc