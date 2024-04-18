# XBNet - Xtremely Boosted Network
## Boosted neural network for tabular data

XBNet, short for Xtremely Boosted Network, is an innovative neural network architecture designed specifically for tabular data analysis. It combines the strengths of traditional neural networks with the interpretability and robustness of gradient boosted trees to create a highly effective and versatile model.

### What XBNet Does
XBNet leverages a novel optimization technique called Boosted Gradient Descent for Tabular Data. This approach enhances the interpretability and performance of neural networks when applied to tabular datasets. By initializing with feature importance from gradient boosted trees and updating weights using both gradient descent and feature importance, XBNet achieves superior performance and training stability compared to traditional neural networks.

### Benefits of Testing on Different Datasets
Testing XBNet on various datasets is crucial for several reasons:

Generalizability: Different datasets exhibit diverse characteristics, such as varying distributions, dimensions, and feature importance. Testing XBNet on multiple datasets allows us to assess its ability to generalize well across different data domains.

Benchmarking: By evaluating XBNet on a range of datasets with known ground truths, we can establish performance benchmarks and compare its effectiveness against other machine learning models.

Identifying Strengths and Weaknesses: Each dataset presents unique challenges and opportunities. Testing XBNet on diverse datasets helps identify its strengths in handling specific data types or tasks, as well as areas where improvement may be needed.

Validation of Claims: Testing XBNet on different datasets provides empirical evidence to validate claims made in research papers or documentation regarding its performance and capabilities.

In summary, testing XBNet on various datasets is essential for assessing its generalizability, benchmarking its performance, identifying strengths and weaknesses, and validating its claims. This approach ensures a thorough understanding of XBNet's effectiveness and suitability for real-world applications across different domains.

---

LINK TO RESEARCH PAPER

[Research Paper](https://arxiv.org/pdf/2106.05239.pdf)

---

### Installation :

To utilize XBNet and replicate the experiments, follow these steps:

1. **Clone Repository:**
```
git clone https://github.com/KaurDiljot/FP.git

```
Install using setup.py:
```
python setup.py install

```
This command will install XBNet and its dependencies on your system.

OR
Install via pip:
```
pip install --upgrade git+https://github.com/tusharsarkar3/XBNet.git
```
This command will directly install XBNet from the GitHub repository.

---

### Example for using
```
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from XBNet.training_utils import training,predict
from XBNet.models import XBNETClassifier
from XBNet.run import run_XBNET

data = pd.read_csv('test\Iris (1).csv')
print(data.shape)
x_data = data[data.columns[:-1]]
print(x_data.shape)
y_data = data[data.columns[-1]]
le = LabelEncoder()
y_data = np.array(le.fit_transform(y_data))
print(le.classes_)

X_train,X_test,y_train,y_test = train_test_split(x_data.to_numpy(),y_data,test_size = 0.3,random_state = 0)
model = XBNETClassifier(X_train,y_train,2)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

m,acc, lo, val_ac, val_lo = run_XBNET(X_train,X_test,y_train,y_test,model,criterion,optimizer,32,300)
print(predict(m,x_data.to_numpy()[0,:]))

import matplotlib.pyplot as plt
plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.plot(acc,label='training accuracy')
plt.plot(val_ac,label = 'validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.grid()
plt.subplot(1,2,2)
plt.plot(lo,label='training loss')
plt.plot(val_lo,label = 'validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend() 
plt.grid()
```
---
### Implemented on Breast Cancer Dataset

To reproduce the results of the research paper, XBNet was implemented on the breast cancer dataset. 

Training Loss after epoch 100 is 0.14333264927069347 and Accuracy is 94.50549450549451

Validation Loss after epoch 100 is 0.08733534067869186 and Accuracy is 95.6140350877193

### Training Performance:
```
              precision    recall  f1-score   support

           0       0.93      0.92      0.93       170
           1       0.95      0.96      0.96       285

    accuracy                           0.95       455
   macro avg       0.94      0.94      0.94       455
weighted avg       0.94      0.95      0.95       455
```
### Validation Performance:
```
              precision    recall  f1-score   support

           0       0.95      0.93      0.94        42
           1       0.96      0.97      0.97        72

    accuracy                           0.96       114
   macro avg       0.96      0.95      0.95       114
weighted avg       0.96      0.96      0.96       114
```
The model achieved a training accuracy of 94.51% and a validation accuracy of 95.61% after 100 epochs.
Precision, recall, and F1-score metrics are provided for both training and validation sets, indicating robust performance across both sets.
The model demonstrates good generalization, as evidenced by comparable performance metrics on the validation set.
These results suggest that the model has learned effectively from the training data and can generalize well to unseen data.

---
### Reference

@misc{sarkar2021xbnet,
      title={XBNet : An Extremely Boosted Neural Network}, 
      author={Tushar Sarkar},
      year={2021},
      eprint={2106.05239},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

---
