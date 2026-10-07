# Persian Text Classification with TF-IDF and LinearSVC

A simple Machine Learning project for **Persian text classification** using **TF-IDF** and **LinearSVC**.

## 📌 About the Project

This project uses a Persian text dataset to train a machine learning model that can classify new text based on patterns learned from the training data.

The project converts text into numerical features using **TF-IDF** and then uses a **LinearSVC** classifier to make predictions.

For example:

```text
بد بود نخرید
```

can be given to the trained model and the model will predict its corresponding category.

---

## 🛠️ Technologies

* Python
* Pandas
* Scikit-learn
* TF-IDF
* LinearSVC
* Jupyter Notebook
* VS Code

---

## 🧠 How It Works

The project follows this process:

```text
Persian Text
     ↓
TF-IDF Vectorization
     ↓
Numerical Features
     ↓
LinearSVC Model
     ↓
Prediction
```

### 1. Load the Dataset

The dataset is loaded using Pandas:

```python
data = pd.read_csv('Dgkala_train.csv')
```

### 2. Select Input and Output

The `Text` column is used as the input and the `Suggestion` column is used as the target:

```python
x = data['Text']
y = data['Suggestion']
```

### 3. Convert Text to Numbers

Machine learning models cannot directly understand text, so TF-IDF is used:

```python
tfidf = TfidfVectorizer()
x_transformed = tfidf.fit_transform(x)
```

### 4. Train the Model

A LinearSVC model is trained using the transformed data:

```python
model = LinearSVC()
model.fit(x_transformed, y)
```

### 5. Make a Prediction

New text can then be classified:

```python
test = 'بد بود نخرید'

prediction = model.predict(
    tfidf.transform([test])
)

print(prediction)
```

---

## ⚠️ Important Note

When using `TfidfVectorizer`, the new text must be passed as a list.

### Correct:

```python
tfidf.transform([test])
```

### Incorrect:

```python
tfidf.transform(test)
```

The brackets are important because `transform()` expects an iterable containing text documents.

---

## 📂 Dataset

The project uses:

```text
Dgkala_train.csv
```

The main columns used by the model are:

| Column       | Description                 |
| ------------ | --------------------------- |
| `Text`       | Persian text used as input  |
| `Suggestion` | Target/classification label |

Make sure the dataset is located in the correct project directory.

---

## 🚀 Installation

First, install the required libraries:

```bash
pip install pandas scikit-learn
```

---

## ▶️ Running the Project

1. Clone or download the project.
2. Make sure `Dgkala_train.csv` is available.
3. Open `main(6).ipynb` using Jupyter Notebook, JupyterLab, or VS Code.
4. Install the required Python libraries.
5. Run the notebook cells from top to bottom.
6. Enter a test sentence and check the model prediction.

---

## 🔮 Future Improvements

Some possible improvements for this project:

* Add a train/test split
* Calculate model accuracy
* Add a classification report
* Test more Persian sentences
* Improve Persian text preprocessing
* Save the trained model
* Save the TF-IDF vectorizer
* Create a graphical user interface
* Create a web interface for predictions

---

## 👨‍💻 Author

**Amirhosein Kamali**

AI Engineer | Python Developer

---

## 📚 Purpose

This project was created for learning and practicing:

* Natural Language Processing (NLP)
* Text Classification
* Machine Learning
* TF-IDF
* Support Vector Machines
* Python
* Scikit-learn

---

## 📄 License

This project is intended for educational and learning purposes.
