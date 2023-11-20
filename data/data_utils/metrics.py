import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt



def auc(y_true, y_pred, plot=False):
	"""
    calculates the area under the ROC the (receiver operating curve) graph of the given data

    :y_ture: true predicition values {-1,1}
    :y_pred: predicted values between -1 and 1
    :plot: plots the ROC graph if true
    :return: area under the ROC graph
    """ 

	o = y_pred.argsort()[::-1]

	pos_step = 1/len([y for y in y_true if y == 1])
	neg_step = 1/len([y for y in y_true if y == -1])

	n = len(y_true)
	fpr = np.zeros(n + 1) 
	tpr = np.zeros(n + 1) 

	auc = 0
	for i in range(1, n+1):
		fpr[i] = fpr[i-1] + (1/2)*(1 - y_true[o[i-1]]) * neg_step
		tpr[i] = tpr[i-1] + (1/2)*(1 + y_true[o[i-1]]) * pos_step

		auc += (fpr[i] - fpr[i-1])* (tpr[i])

	if plot:
		plt.title('Receiver Operating Characteristic (ROC)')
		plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc)
		plt.legend(loc = 'lower right')
		plt.plot([0, 1], [0, 1],'r--')
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')
		plt.show()

	return auc