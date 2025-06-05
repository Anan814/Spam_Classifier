# Custom Logistic and Linear Regression from Scratch

This repository contains a Python implementation of Logistic Regression and Linear Regression algorithms built entirely from scratch, without relying on high-level machine learning libraries like Scikit-learn for the core model logic. It demonstrates the fundamental principles of these algorithms, including gradient descent, data preprocessing, and model evaluation.

---

##  Features

* **From-Scratch Implementation**: Core logic for Logistic and Linear Regression, including cost functions and gradient descent updates, is implemented using only NumPy.
* **Data Preprocessing**: Includes functions for loading CSV datasets, handling categorical variables (one-hot encoding with `pandas.get_dummies`), and feature normalization (standardization).
* **Automatic Model Selection**: The script intelligently determines whether to apply Logistic Regression (for binary classification) or Linear Regression (for continuous regression) based on the target variable's unique values.
* **Cost Function Visualization**: Plots the cost history during training using Matplotlib, allowing you to visualize the convergence of the gradient descent algorithm.
* **Modular Design**: Organized into clear functions and classes for better readability and maintainability.

---

##  Getting Started

To get a copy of the project up and running on your local machine, follow these simple steps.

### Prerequisites

* Python 3.x
* `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Anan814/ML_Asg1.git](https://github.com/Anan814/ML_Asg1.git)
    cd ML_Asg1
    ```
    *(Note: Replace `Anan814/ML_Asg1.git` with your actual repository URL if different.)*

2.  **Install the required Python packages:**
    ```bash
    pip install numpy pandas matplotlib
    ```

---

##  Usage

The main script is designed to be run from your terminal. It expects a dataset file path and the name of the target column.

1.  **Prepare your dataset(s):**
    Ensure you have your CSV files (e.g., `ModifiedHeartDisease.csv`, `CustomerService.csv`) in the same directory as the script, or provide the full path to them.

2.  **Run the script:**
    The example usage is already integrated into the `main.py` file. When you run the script, it will process the specified datasets.

    ```python
    # Example usage within your Python script (e.g., main.py)
    # Ensure this part is present at the end of your code

    # For ModifiedHeartDisease.csv (assuming 'target' is a binary column)
    print("Results for ModifiedHeartDisease.csv:")
    main("ModifiedHeartDisease.csv", "target")

    # For CustomerService.csv (assuming 'Time_to_Resolve' is a continuous column)
    print("\nResults for CustomerService.csv:")
    main("CustomerService.csv", "Time_to_Resolve")
    ```

    Execute your Python file from the terminal:
    ```bash
    python your_script_name.py
    ```

    The script will:
    * Load the specified dataset.
    * Preprocess the data.
    * Train the appropriate regression model.
    * Print evaluation metrics (e.g., **Accuracy** for Logistic Regression, **Mean Squared Error (MSE)** and **R² Score** for Linear Regression).
    * Display a plot of the cost function over iterations.

---

##  Code Structure

The project is structured with clear functions and classes for easy understanding:

* **`load_dataset(file_path)`**: Loads a CSV file into a Pandas DataFrame.
* **`preprocess_data(df, target_column)`**: Separates features and target, handles categorical variables via one-hot encoding, normalizes numerical features, and adds a bias term.
* **`LogisticRegression` Class**: Implements the sigmoid function, binary cross-entropy cost, gradient descent for fitting, and binary prediction.
* **`LinearRegression` Class**: Implements Mean Squared Error (MSE) cost, gradient descent for fitting, and continuous prediction.
* **`train_and_evaluate(X, y)`**: Selects the appropriate regression model, splits data, trains the model, evaluates it with relevant metrics, and plots the cost history.
* **`main(file_path, target_column)`**: The primary function that orchestrates the entire workflow from data loading to evaluation.

---

##  Algorithms Explained

This project provides hands-on implementations of two fundamental supervised learning algorithms and the core optimization technique:

* ### Logistic Regression
    * Used for **binary classification** problems (e.g., predicting `0` or `1`).
    * It models the probability that a given input belongs to a particular class using the **sigmoid function** to map predictions between 0 and 1.
    * Optimized by minimizing the **binary cross-entropy loss** function using **gradient descent**.

* ### Linear Regression
    * Used for **regression** problems (predicting a continuous numerical value).
    * It models the relationship between a dependent variable and one or more independent variables by fitting a linear equation to the observed data.
    * Optimized by minimizing the **Mean Squared Error (MSE)**, which measures the average of the squared differences between predicted and actual values, using **gradient descent**.

* ### Gradient Descent
    * An iterative optimization algorithm used to minimize the cost function of a model.
    * It works by repeatedly adjusting the model's parameters (weights) in the direction opposite to the gradient of the cost function, gradually moving towards the minimum cost.

---

##  Dependencies

* `numpy`
* `pandas`
* `matplotlib`

---

##  Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.

---

##  License

This project is open-source and available under the [MIT License](LICENSE).
