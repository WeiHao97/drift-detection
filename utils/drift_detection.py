#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 21:27:34 2022

@author: lingxiaoli
"""

import numpy as np
from scipy.stats import chi2_contingency, ks_2samp, entropy
from scipy.special import softmax


class ChiSquareDrift:

    def __init__(self, x_ref, threshold: float = .05):
        self.x_ref = self.process_data(x_ref)
        self.threshold = threshold

    def updata_ref(self, x_ref):
        self.x_ref = self.process_data(x_ref)
    
    def update_threshold(self, threshold):
        self.threshold = threshold

    def process_data(self, x):
        margin_width = 0.1
        temp = softmax(x, axis=-1)
        top_2_probs = -np.partition(-temp, kth=1, axis=-1)[:, :2]
        diff = top_2_probs[:, 0] - top_2_probs[:, 1]
        x_logist = (diff < margin_width).astype(int)
        return x_logist[:, None]

    def feature_score_Chi(self, x):
        x = self.process_data(x)
        vals = [0, 1]
        x_ref_count = self.get_counts(self.x_ref, vals)
        x_count = self.get_counts(x, vals)
        p_val = np.zeros(1, dtype=np.float32)
        dist = np.zeros_like(p_val)
        contingency_table = np.vstack((x_ref_count, x_count))
        dist, p_val, _, _ = chi2_contingency(contingency_table)
        return p_val, dist

    def get_counts(self, x, vals):
        return [(x[:] == v).sum() for v in vals]

    def get_result(self, x):
        p_vals, dist = self.feature_score_Chi(x)
        threshold = self.threshold
        drift_pred = int((p_vals < threshold).any())  # type: ignore[assignment]
        cd = {}
        cd['is_drift'] = drift_pred
        cd['p_val'] = p_vals
        cd['threshold'] = threshold
        cd['distance'] = dist
        return cd

class KSDrift:

    def __init__(self, x_ref, threshold: float = .05):
        self.x_ref = self.process_data(x_ref)
        self.threshold = threshold

    def updata_ref(self, x_ref):
        self.x_ref = self.process_data(x_ref)
    
    def update_threshold(self, threshold):
        self.threshold = threshold

    def process_data(self, x):
        return entropy(softmax(x, axis=-1), axis=-1)

    def feature_score_KS(self, x):
        x = self.process_data(x)
        p_val = np.zeros(1, dtype=np.float32)
        dist = np.zeros_like(p_val)
        dist, p_val = ks_2samp(self.x_ref, x, alternative='two-sided', mode='asymp')
        return p_val, dist

    def get_result(self, x):
        p_vals, dist = self.feature_score_KS(x)
        threshold = self.threshold
        drift_pred = int((p_vals < threshold).any())  # type: ignore[assignment]
        cd = {}
        cd['is_drift'] = drift_pred
        cd['p_val'] = p_vals
        cd['threshold'] = threshold
        cd['distance'] = dist
        return cd

def drift_detection(x_ref, threshold: float = .05, method='KSDrift'):
    if method == 'KSDrift':
        return KSDrift(x_ref, threshold)
    elif method == "ChiSquareDrift":
        return ChiSquareDrift(x_ref, threshold)