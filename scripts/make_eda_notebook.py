import json
import os

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 01. Exploratory Data Analysis (EDA) - CLAUDETTE-ToS Dataset\n",
            "\n",
            "This notebook inspects the CLAUDETTE-ToS benchmark dataset:\n",
            "1. **Class distribution** across train, validation, and test splits (fair vs. unfair).\n",
            "2. **Clause text length and word count distributions**.\n",
            "3. **10 representative clauses per class**.\n",
            "4. **Category breakdown** among unfair clauses.\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "\n",
            "# Load processed splits\n",
            "train_df = pd.read_csv('../data/processed/train.csv')\n",
            "val_df = pd.read_csv('../data/processed/val.csv')\n",
            "test_df = pd.read_csv('../data/processed/test.csv')\n",
            "\n",
            "print(f'Train set: {len(train_df):,} clauses')\n",
            "print(f'Validation set: {len(val_df):,} clauses')\n",
            "print(f'Test set: {len(test_df):,} clauses')\n",
            "print(f'Total: {len(train_df) + len(val_df) + len(test_df):,} clauses')\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Class Distribution & Imbalance Check\n",
            "The positive class (potentially unfair / risky clause) represents only ~10-11% of the corpus."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "summary = []\n",
            "for name, df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:\n",
            "    fair = int((df['label'] == 0).sum())\n",
            "    unfair = int((df['label'] == 1).sum())\n",
            "    summary.append({\n",
            "        'Split': name,\n",
            "        'Total': len(df),\n",
            "        'Fair (0)': fair,\n",
            "        'Unfair (1)': unfair,\n",
            "        '% Unfair': f'{unfair / len(df) * 100:.2f}%',\n",
            "        'Imbalance Ratio (Neg:Pos)': f'{fair / unfair:.2f} : 1'\n",
            "    })\n",
            "summary_df = pd.DataFrame(summary)\n",
            "print(summary_df.to_string(index=False))\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Clause Length & Word Count Distributions"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "stats = train_df.groupby('label')['word_count'].describe()\n",
            "stats.index = ['Fair (0)', 'Unfair (1)']\n",
            "print(stats[['count', 'mean', 'std', 'min', '50%', 'max']])\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. 10 Sample Clauses per Class"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('=== 10 FAIR CLAUSES (CLASS 0) ===')\n",
            "for i, row in train_df[train_df['label'] == 0].head(10).reset_index().iterrows():\n",
            "    print(f\"[{i+1}] {row['text'][:120]}...\")\n",
            "\n",
            "print('\\n=== 10 UNFAIR CLAUSES (CLASS 1) ===')\n",
            "for i, row in train_df[train_df['label'] == 1].head(10).reset_index().iterrows():\n",
            "    print(f\"[{i+1}] [{row['category_names_str']}]\")\n",
            "    print(f\"    {row['text'][:120]}...\")\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Breakdown of Categories in Unfair Clauses"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from collections import Counter\n",
            "all_cats = []\n",
            "for cats in train_df[train_df['label'] == 1]['category_names_str'].str.split('; '):\n",
            "    all_cats.extend(cats)\n",
            "cat_counts = pd.Series(Counter(all_cats)).sort_values(ascending=False)\n",
            "cat_df = pd.DataFrame({'Count': cat_counts, 'Percentage (%)': (cat_counts / len(train_df[train_df['label'] == 1]) * 100).round(2)})\n",
            "print(cat_df)\n"
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

os.makedirs("notebooks", exist_ok=True)
with open("notebooks/01_explore_data.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("notebooks/01_explore_data.ipynb successfully generated.")
