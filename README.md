
# 🌸 Iris Species Classification Project 🌸

Welcome to the **Iris Species Classification** project! This project demonstrates how to build simple machine learning models using Logistic Regression and Support Vector Machine (SVM) to classify iris flowers based on their features. 🏵️🌿

## 📂 Dataset
The project uses the famous **Iris Dataset**, which contains information about three species of iris flowers: *Setosa*, *Versicolor*, and *Virginica*. Each data point consists of the following features:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

The dataset is loaded from Google Drive using Google Colab.

## 🛠️ Tools and Libraries Used
- **Python** 🐍
- **Pandas** for data manipulation
- **Matplotlib** & **Seaborn** for data visualization 📊
- **Scikit-learn** for machine learning models 🤖

## 🚀 Getting Started

### 1. Mount Google Drive
To start, mount your Google Drive to access the dataset:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Import Libraries
```python
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
```

### 3. Load the Dataset
```python
path = "/content/drive/MyDrive/DataSet/Iris.csv"
iris = pd.read_csv(path)
iris.head()
```

### 4. Prepare Data
Define the input features `X` and the target labels `y`:
```python
x = iris[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]].values
y = iris[["Species"]].values
```

## 🏋️‍♂️ Model Training

### Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x, y)
print(f"Model Accuracy: {model.score(x, y)}")
```

### Support Vector Machine (SVM)
```python
from sklearn import svm
lmodel = svm.SVC()
lmodel.fit(x, y)
```

## 🔍 Results
Both models have been trained on the dataset to classify the species of iris flowers. You can evaluate the accuracy and compare their performance.

### Example Output
```
Model Accuracy: 0.97
```

## 📊 Visualizations
You can use Matplotlib and Seaborn to create visualizations to better understand the data distribution.

Example:
```python
sns.pairplot(iris, hue="Species")
plt.show()
```

## 📄 License
This project is open-source and available under the MIT License.

## 👨‍💻 Author
- **Keshav Khandelwal**

Feel free to reach out if you have any questions or suggestions! 📧✨
