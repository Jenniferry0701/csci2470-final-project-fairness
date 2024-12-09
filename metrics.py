import numpy as np
from sklearn.metrics import accuracy_score,recall_score,precision_score, f1_score,matthews_corrcoef
from typing import Tuple
import itertools

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
    
    def get_intersectional_groups(self):
        unique_values = [np.unique(attr) for attr in self.protected_attributes]
        groups = list(itertools.product(*unique_values))
        return groups
    
    def get_group_mask(self, group_values):
        mask = np.ones(len(self.y_true), dtype=bool)
        for attr_idx, attr_val in enumerate(group_values):
            mask &= (self.protected_attributes[attr_idx] == attr_val)
        return mask
    
    
    def get_group_metrics(self) -> Tuple[list[dict], list[dict]]:
        # TODO: refactor depending on values of protected_attributes? (not necessarily binary)
        unpriv_metrics_list = []
        priv_metrics_list = []
        for protected_attribute_vals in self.protected_attributes:
            mask_unpriv = protected_attribute_vals == 0
            mask_priv = protected_attribute_vals == 1
            
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

            unpriv_metrics_list.append(unpriv_metrics)
            priv_metrics_list.append(priv_metrics)
        
        return unpriv_metrics_list, priv_metrics_list
    
    def statistical_parity_difference(self) -> list[float]:
        """
        Calculate Statistical Parity Difference (SPD).
        
        SPD = P(Ŷ=1|A=0) - P(Ŷ=1|A=1)
        where Ŷ is the predicted label and A is the protected attribute.
        
        Returns:
            list[float]: SPD values
        """
        unpriv_metrics_list, priv_metrics_list = self.get_group_metrics()
        spd_val_list = []
        for unpriv_metrics, priv_metrics in zip(unpriv_metrics_list, priv_metrics_list):
            prob_pos_unpriv = unpriv_metrics['positive_pred'] / unpriv_metrics['total']
            prob_pos_priv = priv_metrics['positive_pred'] / priv_metrics['total']
            spd_val_list.append(round(prob_pos_unpriv - prob_pos_priv, 2))
        
        return spd_val_list

    def intersectional_spd(self):
        groups = self.get_intersectional_groups()
        spd_vals = []
        
        for i, group_i in enumerate(groups):
            mask_i = self.get_group_mask(group_i)
            if mask_i.sum() == 0:
                continue
            
            prob_pos_i = np.mean(self.y_pred[mask_i])
            
            for j, group_j in enumerate(groups):
                if i == j:
                    continue
                
                mask_j = self.get_group_mask(group_j)
                if mask_j.sum() == 0:
                    continue
                
                prob_pos_j = np.mean(self.y_pred[mask_j])
                spd_vals.append(abs(prob_pos_i - prob_pos_j))
        
        return round(max(spd_vals), 2) if spd_vals else 0
    
    def average_odds_difference(self) -> float:
        """
        Calculate Average Odds Difference (AOD).
        
        AOD = 1/2[(FPR₀ - FPR₁) + (TPR₀ - TPR₁)]
        where FPR is False Positive Rate and TPR is True Positive Rate
        
        Returns:
            list[float]: AOD values for each protected attribute
        """
        unpriv_metrics_list, priv_metrics_list = self.get_group_metrics()
        aod_val_list = []
        for unpriv_metrics, priv_metrics in zip(unpriv_metrics_list, priv_metrics_list):
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
            aod_val_list.append(round(0.5 * ((fpr_unpriv - fpr_priv) + (tpr_unpriv - tpr_priv)), 2))
        
        return aod_val_list
    
    def equal_opportunity_difference(self) -> float:
        """
        Calculate Equal Opportunity Difference (EOD).
        
        EOD = TPR₀ - TPR₁
        where TPR is True Positive Rate
        
        Returns:
            list[float]: EOD values for each protected attribute
        """
        unpriv_metrics_list, priv_metrics_list = self.get_group_metrics()
        eod_val_list = []
        for unpriv_metrics, priv_metrics in zip(unpriv_metrics_list, priv_metrics_list):
            tpr_unpriv = (unpriv_metrics['true_positive'] / 
                        unpriv_metrics['positive_true'] if unpriv_metrics['positive_true'] > 0 else 0)
            tpr_priv = (priv_metrics['true_positive'] / 
                        priv_metrics['positive_true'] if priv_metrics['positive_true'] > 0 else 0)
            eod_val_list.append(round(tpr_unpriv - tpr_priv))
        
        return eod_val_list

class DifferentialFairnessMetrics:
    """
    Implementation of Differential Fairness metrics from Islam et al. (2023).
    DF provides interpretable guarantees on fairness across intersectional groups.
    
    Attributes:
        y_true: Ground truth labels (0 or 1) 
        y_pred: Predicted probabilities or labels (0 or 1)
        protected_attributes: List of arrays containing protected attribute values
    """
    def __init__(self, y_true, y_pred, protected_attributes):
        self.y_true = y_true
        self.y_pred = y_pred 
        self.protected_attributes = protected_attributes
        
        # Dirichlet smoothing parameter
        self.alpha = 1.0
        
    def get_intersectional_groups(self):
        unique_values = [np.unique(attr) for attr in self.protected_attributes]
        groups = list(itertools.product(*unique_values))
        return groups
    
    
    def get_group_mask(self, group_values):
        mask = np.ones(len(self.y_true), dtype=bool)
        for attr_idx, attr_val in enumerate(group_values):
            mask &= (self.protected_attributes[attr_idx] == attr_val)
        return mask
    
    def compute_group_probabilities(self, y, group_mask):
        if group_mask.sum() == 0:
            return 0
            
        count = np.sum(y[group_mask])
        total = group_mask.sum()
        prob = (count + self.alpha) / (total + 2*self.alpha)
        return prob

    def differential_fairness(self):
        """
        Compute the differential fairness metric ε.
        
        ε = max_{y,s_i,s_j} |ln P(y|s_i) - ln P(y|s_j)|
        
        Returns:
            epsilon: The differential fairness score (lower is better)
        """
        groups = self.get_intersectional_groups()
        
        max_epsilon = 0
        
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                group_i_mask = self.get_group_mask(groups[i])
                group_j_mask = self.get_group_mask(groups[j])
                
                if group_i_mask.sum() == 0 or group_j_mask.sum() == 0:
                    continue
                
                p_pos_i = self.compute_group_probabilities(self.y_pred, group_i_mask) 
                p_pos_j = self.compute_group_probabilities(self.y_pred, group_j_mask)
                p_neg_i = 1 - p_pos_i
                p_neg_j = 1 - p_pos_j
                
                if p_pos_i > 0 and p_pos_j > 0:
                    eps_pos = abs(np.log(p_pos_i) - np.log(p_pos_j))
                else:
                    eps_pos = 0
                    
                if p_neg_i > 0 and p_neg_j > 0:
                    eps_neg = abs(np.log(p_neg_i) - np.log(p_neg_j))
                else:
                    eps_neg = 0
                    
                epsilon = max(eps_pos, eps_neg)
                max_epsilon = max(max_epsilon, epsilon)
                
        return max_epsilon

    def bias_amplification(self, data_epsilon):
        """
        Compute the DF bias amplification measure.
        
        Args:
            data_epsilon: The DF score of the training data
            
        Returns:
            bias_amp: ε_model - ε_data (negative values indicate reduction in bias)
        """
        model_epsilon = self.differential_fairness()
        return model_epsilon - data_epsilon
    
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
    df_metrics = DifferentialFairnessMetrics(y_true, y_pred, protected_attributes)
    data_df = DifferentialFairnessMetrics(y_true, y_true, protected_attributes)
    data_epsilon = data_df.differential_fairness()
    return {
        'accuracy': metrics.accuracy(),
        'precision': metrics.precision(),
        'recall': metrics.recall(),
        'f1': metrics.f1(),
        'statistical_parity_difference': np.mean(metrics.statistical_parity_difference()),
        'intersectional_spd': metrics.intersectional_spd(),
        'average_odds_difference': np.mean(metrics.average_odds_difference()),
        'equal_opportunity_difference': np.mean(metrics.equal_opportunity_difference()),
        'differential_fairness': df_metrics.differential_fairness(),
        'df_bias_amplification': df_metrics.bias_amplification(data_epsilon)
    }