# to draw roc curve

import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 读取数据 read the data
df = pd.read_csv('all_Ensemble_0_azeotrope_classification_cleaned_AGCNetCLF_azeotrope_classification_cleaned_915_2025-10-24-08-21.csv') # result .csv

y_true = df['Target'].values
y_scores = df['Predict'].values

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

roc_data = pd.DataFrame({
    'FPR': fpr,
    'TPR': tpr,
    'Threshold': thresholds
})

distances = np.sqrt(fpr**2 + (1-tpr)**2)
best_idx = np.argmin(distances)
best_threshold = thresholds[best_idx]
best_fpr = fpr[best_idx]
best_tpr = tpr[best_idx]

summary_data = pd.DataFrame({
    'Metric': ['AUC', 'Best_Threshold', 'Best_FPR', 'Best_TPR'],
    'Value': [roc_auc, best_threshold, best_fpr, best_tpr]
})

with open('roc_curve_results.csv', 'w', newline='', encoding='utf-8') as f:
    summary_data.to_csv(f, index=False)
    
    f.write('\n')
    
    roc_data.to_csv(f, index=False)

print("ROC results have been saved to 'roc_curve_results.csv'")
print(f"AUC = {roc_auc:.6f}")
print(f"best threshold = {best_threshold:.8f}")
print(f"bese FPR = {best_fpr:.6f}")
print(f"best TPR = {best_tpr:.6f}")

# 绘制ROC曲线
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='random classifier')
plt.plot(best_fpr, best_tpr, 'ro', markersize=10, label=f'best threshold point\nthreshold={best_threshold:.4f}')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve - azeotropic classification')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('roc_curve_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print("ROC have been saved to 'roc_curve_plot.png'")