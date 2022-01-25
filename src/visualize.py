import pandas as pd
import talib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot
import numpy as np

from src.indicators import get_price_patterns

def visualize_feature_importance(df_feature_importance):

    df_feature_importance.sort_values(by='Importance', inplace=True, ascending=False)

    fig, ax = plt.subplots(figsize=(24, 48))
    sns.barplot(x='Importance', y=df_feature_importance.index, data=df_feature_importance, palette='Blues_d', ax=ax)
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_title(f'Feature Importance', size=20, pad=20)
    plt.show()


def visualize_predictions(train_labels, train_predictions, test_predictions):

    fig, axes = plt.subplots(ncols=2, figsize=(24, 6))
    sns.scatterplot(train_labels, train_predictions, ax=axes[0])
    sns.distplot(train_predictions, label='Train Predictions', ax=axes[1])
    sns.distplot(test_predictions, label='Test Predictions', ax=axes[1])

    axes[0].set_xlabel(f'Train Labels', size=18)
    axes[0].set_ylabel(f'Train Predictions', size=18)
    axes[1].set_xlabel('')
    axes[1].legend(prop={'size': 18})
    for i in range(2):
        axes[i].tick_params(axis='x', labelsize=15)
        axes[i].tick_params(axis='y', labelsize=15)
    axes[0].set_title(f'Train Labels vs Train Predictions', size=20, pad=20)
    axes[1].set_title(f'Train and Test Predictions\' Distributions', size=20, pad=20)

    plt.show()


def visualize_target(df, feature):

    print(f'{feature}\n{"-" * len(feature)}')

    print(f'Mean: {df[feature].mean():.4f}  -  Median: {df[feature].median():.4f}  -  Std: {df[feature].std():.4f}')
    print(f'Min: {df[feature].min():.4f}  -  25%: {df[feature].quantile(0.25):.4f}  -  50%: {df[feature].quantile(0.5):.4f}  -  75%: {df[feature].quantile(0.75):.4f}  -  Max: {df[feature].max():.4f}')
    print(f'Skew: {df[feature].skew():.4f}  -  Kurtosis: {df[feature].kurtosis():.4f}')
    missing_count = df[df[feature].isnull()].shape[0]
    total_count = df.shape[0]
    print(f'Missing Values: {missing_count}/{total_count} ({missing_count * 100 / total_count:.4f}%)')

    fig, axes = plt.subplots(ncols=2, figsize=(24, 6), dpi=100)

    sns.distplot(df[feature], label=feature, ax=axes[0])
    axes[0].axvline(df[feature].mean(), label='Mean', color='r', linewidth=2, linestyle='--')
    axes[0].axvline(df[feature].median(), label='Median', color='b', linewidth=2, linestyle='--')
    axes[0].legend(prop={'size': 15})
    probplot(df[feature], plot=axes[1])

    for i in range(2):
        axes[i].tick_params(axis='x', labelsize=12.5)
        axes[i].tick_params(axis='y', labelsize=12.5)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
    axes[0].set_title(f'{feature} Distribution', fontsize=15, pad=12)
    axes[1].set_title(f'{feature} Probability Plot', fontsize=15, pad=12)
    plt.show()


def visualize_continuous_feature(df, df_test, feature):

    print(f'{feature}\n{"-" * len(feature)}')

    print(f'Training Mean: {df[feature].mean():.4f}  - Median: {df[feature].median():.4f} - Std: {df[feature].std():.4f}')
    print(f'Test Mean: {df_test[feature].mean():.4f}  - Median: {df_test[feature].median():.4f} - Std: {df_test[feature].std():.4f}')
    print(f'Training Min: {df[feature].min():.4f}  - Max: {df[feature].max():.4f}')
    print(f'Test Min: {df_test[feature].min():.4f}  - Max: {df_test[feature].max():.4f}')
    print(f'Training Skew: {df[feature].skew():.4f}  - Kurtosis: {df[feature].kurtosis():.4f}')
    print(f'Test Skew: {df_test[feature].skew():.4f}  - Kurtosis: {df_test[feature].kurtosis():.4f}')
    training_missing_count = df[df[feature].isnull()].shape[0]
    test_missing_count = df_test[df_test[feature].isnull()].shape[0]
    training_total_count = df.shape[0]
    test_total_count = df_test.shape[0]
    print(f'Training Missing Values: {training_missing_count}/{training_total_count} ({training_missing_count * 100 / training_total_count:.4f}%)')
    print(f'Test Missing Values: {test_missing_count}/{test_total_count} ({test_missing_count * 100 / test_total_count:.4f}%)')

    fig, axes = plt.subplots(ncols=2, figsize=(24, 6), dpi=100, constrained_layout=True)

    # Continuous Feature Training and Test Set Distribution
    sns.distplot(df[feature], label='Training', ax=axes[0])
    sns.distplot(df_test[feature], label='Test', ax=axes[0])
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', labelsize=12.5)
    axes[0].tick_params(axis='y', labelsize=12.5)
    axes[0].legend(prop={'size': 15})
    axes[0].set_title(f'{feature} Distribution in Training and Test Set', fontsize=15, pad=12)

    # Continuous Feature vs target
    sns.scatterplot(df[feature], df['output_gen'], ax=axes[1])
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')
    axes[1].tick_params(axis='x', labelsize=12.5)
    axes[1].tick_params(axis='y', labelsize=12.5)
    axes[1].set_title(f'{feature} vs output_gen', fontsize=15, pad=12)

    plt.show()


def plot_patterns(data):
    """
    Recognized price patterns and plots results. If called not within IPython, you might want to call `plt.show()`
    after calling this method.

    :param data DataFrame with ticks. Could be with or without embed transactions.
    """
    patterns = get_price_patterns(data)
    plt.imshow(patterns.T, aspect='auto')
    plt.ylabel('Patterns')
    plt.yticks(np.arange(len(patterns.columns)), patterns.columns)
    plt.xlabel('Timestamps')
    plt.xticks(np.arange(len(patterns.index))[::100], patterns.index[::100])
    plt.title('Detected price patterns')
    plt.colorbar()