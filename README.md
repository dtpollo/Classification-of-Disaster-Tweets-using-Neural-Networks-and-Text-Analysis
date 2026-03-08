## Notebook Explanation
### Section 1 - Data Exploration
In this section we load the data from the CSV files and check the size of each set.
We also create visualizations to better understand the dataset, including:
- Number of words per tweet
- Number of unique words per tweet
- Average word length per tweet
- Number of characters per tweet
- Length of the last word per tweet
- Number of URLs per tweet
- Average number of characters per word
- Number of punctuation marks per tweet
- Number of hashtags per tweet
- Number of @mentions per tweet
- Number of stop words per tweet

Each visualization compares disaster tweets vs non-disaster tweets to find patterns.

### Section 2 - Text Cleaning
First we create visualization functions to understand the most frequent stop words, 
URLs, mentions, and bigrams in the dataset.

Then we apply three cleaning configurations to prepare the text for the model:

- **Config 1:** Lowercase, remove URLs, remove mentions, remove punctuation
- **Config 2:** Config 1 + remove stop words
- **Config 3:** Config 1 + apply stemming (reduce words to their root form)

### Section 3 - Neural Network Implementation
**Vectorization:**
We test two types of vectorization using TF-IDF:
- **TV1:** Uses the full vocabulary from the dataset
- **TV2:** Uses only the top 5000 most frequent words and bigrams

**Model Architecture:**
All models share the same structure regardless of vectorization:
- Input layer
- Dense layer: 256 neurons, ReLU activation
- Dense layer: 128 neurons, ReLU activation
- Dense layer: 64 neurons, ReLU activation
- Output layer: 1 neuron, Sigmoid activation (binary classification)

We use ReLU to avoid the vanishing gradient problem, and Sigmoid because this is a binary classification problem (disaster or not).
We also use Dropout (0.3) after each hidden layer, which randomly turns off 30% of neurons during training to prevent overfitting.

**Model Compile:**
We compile each model using:
- Loss function: `binary_crossentropy` (standard for binary problems)
- Metrics: Accuracy and Precision

**Fit and Metrics:**
We train each model and save the weights to reuse them later without retraining.
After training we calculate Accuracy, F1-score and Confusion Matrix for each model.

### Section 4 - Results Comparison
Here we plot all model results in a single bar chart to easily compare 
Accuracy and F1-score across all configurations and vectorizations.

### Section 5 - Cleaning Analysis
Here we answer the questions from Section 2 about the impact of each cleaning technique,
and present a comparative table with all results.

### Section 6 - Real-Time Prediction
Here we reload the best model from each vectorization using the saved weights,
without needing to retrain. We also save the vectorizer to a file so we don't need 
to redefine it again.

For **TV1** the best model was **Config 3** (Accuracy: 0.7571, F1: 0.6997).
For **TV2** the best model was **Config 1** (Accuracy: 0.7735, F1: 0.7298).

The user can type any tweet and the function will automatically clean the text,
vectorize it, and show the prediction and confidence score for both models.
