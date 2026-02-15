"""
Data Preparation Script
Collects and formats Python/ML Q&A data for training
"""

import json
import os
from typing import List, Dict

def create_sample_dataset() -> List[Dict[str, str]]:
    """
    Creates a sample dataset of Python/ML Q&A pairs.
    In production, you would:
    1. Scrape Stack Overflow (with rate limiting and API)
    2. Use existing datasets from Hugging Face
    3. Curate your own high-quality examples
    """
    
    sample_data = [
        {
            "instruction": "What is the difference between a list and a tuple in Python?",
            "response": "Lists and tuples are both sequence types in Python, but they have key differences:\n\n1. **Mutability**: Lists are mutable (can be modified after creation), while tuples are immutable (cannot be changed).\n2. **Syntax**: Lists use square brackets [], tuples use parentheses ().\n3. **Performance**: Tuples are slightly faster than lists due to immutability.\n4. **Use case**: Use lists when you need to modify data, tuples when data should remain constant.\n\nExample:\n```python\nmy_list = [1, 2, 3]\nmy_list[0] = 10  # Works\n\nmy_tuple = (1, 2, 3)\nmy_tuple[0] = 10  # Raises TypeError\n```"
        },
        {
            "instruction": "How do you handle missing values in a pandas DataFrame?",
            "response": "There are several ways to handle missing values in pandas:\n\n1. **Drop missing values**:\n```python\ndf.dropna()  # Drop rows with any NaN\ndf.dropna(axis=1)  # Drop columns with any NaN\n```\n\n2. **Fill missing values**:\n```python\ndf.fillna(0)  # Replace with 0\ndf.fillna(df.mean())  # Replace with mean\ndf.fillna(method='ffill')  # Forward fill\n```\n\n3. **Check for missing values**:\n```python\ndf.isnull().sum()  # Count NaN per column\ndf.isna().any()  # Check if any NaN exists\n```\n\n4. **Interpolate**:\n```python\ndf.interpolate()  # Linear interpolation\n```\n\nChoose the method based on your data and use case."
        },
        {
            "instruction": "Explain the difference between supervised and unsupervised learning.",
            "response": "Supervised and unsupervised learning are two main categories of machine learning:\n\n**Supervised Learning**:\n- Trains on labeled data (input-output pairs)\n- Goal: Learn a mapping from inputs to outputs\n- Examples: Classification, regression\n- Algorithms: Linear regression, decision trees, neural networks\n- Use case: Predicting house prices, image classification\n\n**Unsupervised Learning**:\n- Trains on unlabeled data (only inputs)\n- Goal: Discover patterns or structure in data\n- Examples: Clustering, dimensionality reduction\n- Algorithms: K-means, PCA, autoencoders\n- Use case: Customer segmentation, anomaly detection\n\n**Key difference**: Supervised learning has target labels, unsupervised learning finds hidden patterns without labels."
        },
        {
            "instruction": "How do you create a virtual environment in Python?",
            "response": "There are several ways to create a virtual environment in Python:\n\n**1. Using venv (built-in, Python 3.3+)**:\n```bash\n# Create virtual environment\npython -m venv myenv\n\n# Activate (Windows)\nmyenv\\Scripts\\activate\n\n# Activate (macOS/Linux)\nsource myenv/bin/activate\n\n# Deactivate\ndeactivate\n```\n\n**2. Using virtualenv**:\n```bash\npip install virtualenv\nvirtualenv myenv\nsource myenv/bin/activate\n```\n\n**3. Using conda**:\n```bash\nconda create -n myenv python=3.9\nconda activate myenv\nconda deactivate\n```\n\nVirtual environments isolate project dependencies and prevent conflicts between different projects."
        },
        {
            "instruction": "What is overfitting in machine learning and how can you prevent it?",
            "response": "**Overfitting** occurs when a model learns the training data too well, including noise and outliers, leading to poor performance on new data.\n\n**Signs of overfitting**:\n- High training accuracy but low validation/test accuracy\n- Model performs well on training set but poorly on unseen data\n- Large gap between training and validation loss\n\n**Prevention techniques**:\n\n1. **More training data**: Helps model generalize better\n2. **Cross-validation**: Use k-fold CV to evaluate performance\n3. **Regularization**: Add penalty for complexity (L1, L2)\n4. **Dropout**: Randomly drop neurons during training (neural networks)\n5. **Early stopping**: Stop training when validation loss stops improving\n6. **Simpler model**: Reduce model complexity\n7. **Feature selection**: Remove irrelevant features\n8. **Data augmentation**: Create synthetic training examples\n\nExample with regularization:\n```python\nfrom sklearn.linear_model import Ridge\nmodel = Ridge(alpha=1.0)  # L2 regularization\n```"
        },
        {
            "instruction": "How do you read a CSV file in Python using pandas?",
            "response": "You can read CSV files in pandas using `pd.read_csv()`:\n\n**Basic usage**:\n```python\nimport pandas as pd\n\ndf = pd.read_csv('file.csv')\n```\n\n**Common parameters**:\n```python\ndf = pd.read_csv(\n    'file.csv',\n    sep=',',              # Delimiter\n    header=0,             # Row to use as column names\n    index_col=0,          # Column to use as index\n    usecols=['A', 'B'],   # Select specific columns\n    skiprows=5,           # Skip first 5 rows\n    nrows=100,            # Read only 100 rows\n    na_values=['NA'],     # Additional NA values\n    dtype={'col': int},   # Specify column types\n    parse_dates=['date'], # Parse date columns\n    encoding='utf-8'      # File encoding\n)\n```\n\n**For large files**:\n```python\n# Read in chunks\nfor chunk in pd.read_csv('large.csv', chunksize=10000):\n    process(chunk)\n```"
        },
        {
            "instruction": "What is the difference between a shallow copy and a deep copy in Python?",
            "response": "**Shallow Copy**:\n- Creates a new object but references the same nested objects\n- Changes to nested objects affect both copies\n- Methods: `copy.copy()`, `list.copy()`, `dict.copy()`\n\n**Deep Copy**:\n- Creates a new object and recursively copies all nested objects\n- Complete independence between original and copy\n- Method: `copy.deepcopy()`\n\n**Example**:\n```python\nimport copy\n\noriginal = [[1, 2], [3, 4]]\n\n# Shallow copy\nshallow = copy.copy(original)\nshallow[0][0] = 99\nprint(original)  # [[99, 2], [3, 4]] - modified!\n\n# Deep copy\noriginal = [[1, 2], [3, 4]]\ndeep = copy.deepcopy(original)\ndeep[0][0] = 99\nprint(original)  # [[1, 2], [3, 4]] - unchanged\n```\n\n**Rule of thumb**: Use shallow copy for simple objects, deep copy when dealing with nested structures."
        },
        {
            "instruction": "Explain what MLOps is and why it's important.",
            "response": "**MLOps (Machine Learning Operations)** is a set of practices that combines ML, DevOps, and data engineering to deploy and maintain ML models in production reliably and efficiently.\n\n**Key components**:\n1. **Version control**: Track code, data, and models (Git, DVC)\n2. **CI/CD pipelines**: Automate training, testing, deployment\n3. **Monitoring**: Track model performance, data drift\n4. **Reproducibility**: Ensure experiments can be replicated\n5. **Automation**: Automate retraining and deployment\n\n**Why it's important**:\n- **Reliability**: Ensures models work consistently in production\n- **Scalability**: Deploy models at scale efficiently\n- **Collaboration**: Teams can work together effectively\n- **Maintenance**: Easier to update and retrain models\n- **Governance**: Track model lineage and compliance\n\n**Common tools**:\n- MLflow, Kubeflow, Weights & Biases\n- Docker, Kubernetes\n- GitHub Actions, Jenkins\n- Prometheus, Grafana\n\nMLOps bridges the gap between ML development and production deployment."
        },
        {
            "instruction": "How do you split data into training and testing sets in Python?",
            "response": "Use scikit-learn's `train_test_split()` function:\n\n**Basic usage**:\n```python\nfrom sklearn.model_selection import train_test_split\n\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, \n    test_size=0.2,      # 20% for testing\n    random_state=42     # For reproducibility\n)\n```\n\n**Advanced options**:\n```python\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y,\n    test_size=0.2,\n    random_state=42,\n    shuffle=True,           # Shuffle before split\n    stratify=y              # Maintain class distribution\n)\n```\n\n**Three-way split (train/val/test)**:\n```python\n# First split: 60% train, 40% temp\nX_train, X_temp, y_train, y_temp = train_test_split(\n    X, y, test_size=0.4, random_state=42\n)\n\n# Second split: 20% val, 20% test (from the 40% temp)\nX_val, X_test, y_val, y_test = train_test_split(\n    X_temp, y_temp, test_size=0.5, random_state=42\n)\n```\n\nCommon split ratios: 80/20, 70/30, or 60/20/20 for train/val/test."
        },
        {
            "instruction": "What are Python decorators and how do you use them?",
            "response": "**Decorators** are functions that modify the behavior of other functions or classes without changing their source code.\n\n**Basic structure**:\n```python\ndef my_decorator(func):\n    def wrapper(*args, **kwargs):\n        # Code before function call\n        result = func(*args, **kwargs)\n        # Code after function call\n        return result\n    return wrapper\n\n@my_decorator\ndef my_function():\n    print(\"Hello\")\n```\n\n**Common use cases**:\n\n1. **Timing decorator**:\n```python\nimport time\n\ndef timer(func):\n    def wrapper(*args, **kwargs):\n        start = time.time()\n        result = func(*args, **kwargs)\n        end = time.time()\n        print(f\"{func.__name__} took {end-start:.2f}s\")\n        return result\n    return wrapper\n\n@timer\ndef slow_function():\n    time.sleep(2)\n```\n\n2. **Authentication decorator**:\n```python\ndef require_auth(func):\n    def wrapper(*args, **kwargs):\n        if not user.is_authenticated:\n            raise PermissionError(\"Not authenticated\")\n        return func(*args, **kwargs)\n    return wrapper\n```\n\nDecorators enable clean, reusable code for cross-cutting concerns like logging, caching, and validation."
        }
    ]
    
    return sample_data


def add_your_own_examples() -> List[Dict[str, str]]:
    """
    Template for adding your own custom Q&A pairs.
    Replace these with real questions and answers from your domain.
    """
    custom_data = [
        {
            "instruction": "Your question here",
            "response": "Your detailed answer here"
        },
        # Add more examples...
    ]
    return custom_data


def format_data_for_training(data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Format data into the structure expected by the training script.
    """
    formatted_data = []
    for item in data:
        formatted_data.append({
            "instruction": item["instruction"],
            "response": item["response"]
        })
    return formatted_data


def save_dataset(data: List[Dict[str, str]], output_path: str = "./data/training_data.json"):
    """
    Save the dataset to a JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Dataset saved to {output_path}")
    print(f"📊 Total examples: {len(data)}")


def main():
    """
    Main function to prepare training data.
    """
    print("🔍 Preparing training data...")
    
    # Get sample dataset
    sample_data = create_sample_dataset()
    
    # You can add your own examples
    # custom_data = add_your_own_examples()
    # all_data = sample_data + custom_data
    
    all_data = sample_data
    
    # Format data
    formatted_data = format_data_for_training(all_data)
    
    # Save to file
    save_dataset(formatted_data)
    
    print("\n💡 Tips for collecting more data:")
    print("1. Scrape Stack Overflow with their API")
    print("2. Use existing datasets from Hugging Face")
    print("3. Curate examples from Python documentation")
    print("4. Add your own domain-specific Q&A pairs")
    print("5. Use GPT-4 to generate synthetic examples (with review)")
    print("\n🎯 Aim for at least 1,000-5,000 high-quality examples for good results")


if __name__ == "__main__":
    main()
