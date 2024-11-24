import numpy as np
from sklearn.metrics import accuracy_score,recall_score,precision_score, f1_score,matthews_corrcoef
from typing import Tuple

class FairnessMetrics:
    """
    A class to compute various fairness metrics for binary classification.
    
    Attributes:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        protected_attributes: Binary protected attribute values (0 or 1)
    """
    
    def __init__(self, y_true, y_pred, protected_attributes
    ):
        """
        Initialize the FairnessMetrics class.
        
        Args:
            y_true: Array of true labels
            y_pred: Array of predicted labels
            protected_attributes: Array of protected attribute values
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.protected_attributes = protected_attributes
    
    def accuracy(self, group_mask=None) -> float:
        if group_mask is None:
            y_true = self.y_true
            y_pred = self.y_pred
        else:
            y_true = self.y_true[group_mask]
            y_pred = self.y_pred[group_mask]
        return accuracy_score(y_true, y_pred)
    
    def precision(self, group_mask=None) -> float:
        if group_mask is None:
            y_true = self.y_true
            y_pred = self.y_pred
        else:
            y_true = self.y_true[group_mask]
            y_pred = self.y_pred[group_mask]
        return precision_score(y_true, y_pred,  zero_division=0)
    
    def recall(self, group_mask=None) -> float:
        if group_mask is None:
            y_true = self.y_true
            y_pred = self.y_pred
        else:
            y_true = self.y_true[group_mask]
            y_pred = self.y_pred[group_mask]
        return recall_score(y_true, y_pred,  zero_division=0)

    def f1(self, group_mask=None) -> float:
        if group_mask is None:
            y_true = self.y_true
            y_pred = self.y_pred
        else:
            y_true = self.y_true[group_mask]
            y_pred = self.y_pred[group_mask]
        return f1_score(y_true, y_pred, zero_division=0)
    
    def get_group_metrics(self) -> Tuple[dict, dict]:
        # TODO: refactor depending on values of protected_attributes? (not necessarily binary)
        mask_unpriv = self.protected_attributes == 0
        mask_priv = self.protected_attributes == 1
        
        unpriv_metrics = {
            'total': sum(mask_unpriv),
            'positive_pred': sum(self.y_pred[mask_unpriv]),
            'true_positive': sum((self.y_pred == 1) & (self.y_true == 1) & mask_unpriv),
            'false_positive': sum((self.y_pred == 1) & (self.y_true == 0) & mask_unpriv),
            'positive_true': sum(self.y_true[mask_unpriv])
        }
        
        priv_metrics = {
            'total': sum(mask_priv),
            'positive_pred': sum(self.y_pred[mask_priv]),
            'true_positive': sum((self.y_pred == 1) & (self.y_true == 1) & mask_priv),
            'false_positive': sum((self.y_pred == 1) & (self.y_true == 0) & mask_priv),
            'positive_true': sum(self.y_true[mask_priv])
        }
        
        return unpriv_metrics, priv_metrics
    
    def statistical_parity_difference(self) -> float:
        """
        Calculate Statistical Parity Difference (SPD).
        
        SPD = P(Ŷ=1|A=0) - P(Ŷ=1|A=1)
        where Ŷ is the predicted label and A is the protected attribute.
        
        Returns:
            float: SPD value
        """
        unpriv_metrics, priv_metrics = self.get_group_metrics()
        prob_pos_unpriv = unpriv_metrics['positive_pred'] / unpriv_metrics['total']
        prob_pos_priv = priv_metrics['positive_pred'] / priv_metrics['total']
        
        return prob_pos_unpriv - prob_pos_priv
    
    def average_odds_difference(self) -> float:
        """
        Calculate Average Odds Difference (AOD).
        
        AOD = 1/2[(FPR₀ - FPR₁) + (TPR₀ - TPR₁)]
        where FPR is False Positive Rate and TPR is True Positive Rate
        
        Returns:
            float: AOD value
        """
        unpriv_metrics, priv_metrics = self.get_group_metrics()
        
        tpr_unpriv = (unpriv_metrics['true_positive'] / 
                      unpriv_metrics['positive_true'] if unpriv_metrics['positive_true'] > 0 else 0)
        fpr_unpriv = (unpriv_metrics['false_positive'] / 
                      (unpriv_metrics['total'] - unpriv_metrics['positive_true'])
                      if (unpriv_metrics['total'] - unpriv_metrics['positive_true']) > 0 else 0)
        
        tpr_priv = (priv_metrics['true_positive'] / 
                    priv_metrics['positive_true'] if priv_metrics['positive_true'] > 0 else 0)
        fpr_priv = (priv_metrics['false_positive'] / 
                    (priv_metrics['total'] - priv_metrics['positive_true'])
                    if (priv_metrics['total'] - priv_metrics['positive_true']) > 0 else 0)
        
        return 0.5 * ((fpr_unpriv - fpr_priv) + (tpr_unpriv - tpr_priv))
    
    def equal_opportunity_difference(self) -> float:
        """
        Calculate Equal Opportunity Difference (EOD).
        
        EOD = TPR₀ - TPR₁
        where TPR is True Positive Rate
        
        Returns:
            float: EOD value
        """
        unpriv_metrics, priv_metrics = self.get_group_metrics()
        
        tpr_unpriv = (unpriv_metrics['true_positive'] / 
                      unpriv_metrics['positive_true'] if unpriv_metrics['positive_true'] > 0 else 0)
        tpr_priv = (priv_metrics['true_positive'] / 
                    priv_metrics['positive_true'] if priv_metrics['positive_true'] > 0 else 0)
        
        return tpr_unpriv - tpr_priv

def compute_all_metrics(y_true, y_pred, protected_attributes) -> dict:
    """
    Compute all fairness metrics at once.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        protected_attributes: Protected attribute values (0 or 1)
        
    Returns:
        Dictionary containing all fairness metrics
    """
    metrics = FairnessMetrics(y_true, y_pred, protected_attributes)
    return {
        'accuracy': metrics.accuracy(),
        'precision': metrics.precision(),
        'recall': metrics.recall(),
        'f1': metrics.f1(),
        'statistical_parity_difference': metrics.statistical_parity_difference(),
        'average_odds_difference': metrics.average_odds_difference(),
        'equal_opportunity_difference': metrics.equal_opportunity_difference()
    }