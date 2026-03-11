---
title: Policy Reinstatement Prediction with Regularized Two-Stage Residual Learning
description: A cost-sensitive two-stage ensemble for reinstatement prediction on heavily
  imbalanced data, with a formal comparison of OLS, Ridge, and Lasso objective functions
  and their implications for correlated insurance features.
author: Jiheon (Jay) Park
date: '2026-03-06'
tags:
- Machine Learning
- Data Engineering
- Class Imbalance
- Regularization
- AI Workflow
draft: false
---

## Problem Statement

When an insurance policy lapses due to missed premiums, the customer has a limited window to reinstate it. The business runs monthly outbound campaigns targeting recently lapsed customers, but the reinstatement rate is very low — the vast majority of outreach is directed at customers who will not return.

The task: predict which recently lapsed customers will reinstate within a limited window from the campaign reference date. The constraint: severe class imbalance and a target population restricted to customers who lapsed within a recent lookback period.

Customers who do not reinstate within this window may later enter the [[portfolio/sales-ab-lapsed-customer-reactivation/index|lapsed customer reactivation pipeline]], which uses a persona-based recommendation system to re-engage them with new product offers.

## Feature Engineering

Dozens of features derived from contract records, customer profiles, payment history, loan utilization, and reinstatement records. The engineering involved multiple PySpark scripts joining multiple source tables, each with strict temporal boundaries.

### Time-Travel Consistency

The reference date is set to the last day of the month *before* the campaign month. If the campaign runs in October, features reflect data through September 30th. This prevents leakage from the prediction window.

Several features from the reinstatement table were removed during development because they encoded post-campaign outcomes: unpaid premium counts, reinstatement status codes, and receipt completion indicators. These would have substantially inflated discrimination metrics but are unavailable at prediction time.

### Feature Groups

**Contract Tenure and Structure** — months since policy inception (computed from the reference date, not month-end — a subtle fix that eliminated a 1–30 day temporal error), policy period duration (log-transformed), applied premium aggregates, product type indicators

**Customer Financial Profile** — active and total policy counts (99th-percentile clipped), monthly premium aggregates, a composite customer lifetime value score. The value score is quantile-binned into tiers, with a binary presence flag for the $> 95\%$ of observations where the raw score is zero.

**Lapse Characteristics** — days elapsed since lapse (robust-scaled via $(x - \tilde{x}) / \text{IQR}$, filtered to within the lookback window), lapse timing features (month, day-of-week)

**Loan and Payment Behavior** — loan utilization counts, cumulative loan amounts, payment method transition frequency, policy loan indicators. The automatic premium loan flag is a proxy for cash flow stress: the insurer draws against the policy's cash surrender value to cover premiums, indicating the customer cannot pay out-of-pocket.

**Reinstatement History** — prior reinstatement counts (computed only from events before the reference date). This feature dominates in importance (dominant in a random forest diagnostic check), but it is legitimate: customers who have reinstated before are behaviorally distinct from first-time lapsers.

**Demographic and Channel** — occupational classification (target-encoded with Bayesian smoothing, $\alpha = 20$, for high-cardinality categories), distribution channel (target-encoded, $\alpha = 10$), gender (one-hot), age (with validity flag for zero-value detection and median imputation)

> **Margin note:**
> Bayesian smoothing is equivalent to a Beta prior on the category-level mean. The smoothing parameter $\alpha$ controls the prior strength: larger $\alpha$ pulls rare categories more aggressively toward the global mean.

**Target encoding** uses Bayesian smoothing to prevent overfitting on rare categories:

$$
\hat{\mu}_k = \frac{n_k \cdot \bar{y}_k + \alpha \cdot \bar{y}_{\text{global}}}{n_k + \alpha}
$$

where $n_k$ is the category count, $\bar{y}_k$ the within-category target mean, and $\alpha$ the smoothing parameter. This converges to the global mean for rare categories and to the category-specific mean for well-populated ones.

### Preprocessing

Heavy right skew in premium features (skewness exceeding 800 in extreme cases) required a two-step treatment: 99th-percentile capping followed by $\log(1 + x)$ transformation. Features with $> 95\%$ zero prevalence were converted to binary presence indicators plus log-transformed continuous components. Temporal features were robust-scaled. Missing numerics filled with 0 (encoding absence of activity); missing categoricals filled with "Unknown."

## Objective Functions: OLS, Ridge, and Lasso

> **Margin note:**
> The bias-variance tradeoff is the central tension. OLS has zero bias but maximal variance under collinearity. Ridge and Lasso introduce bias to reduce variance. The optimal point on this tradeoff depends on the feature correlation structure.

The reinstatement model provides a natural setting to examine the relationship between ordinary least squares and its regularized variants — and why the choice of regularization matters when features are correlated.

### Ordinary Least Squares (OLS)

OLS minimizes the unpenalized residual sum of squares:

$$
\hat{\boldsymbol{\beta}}_{\text{OLS}} = \arg\min_{\boldsymbol{\beta}} \sum_{i=1}^{N} (y_i - \mathbf{x}_i^\top \boldsymbol{\beta})^2
$$

OLS is unbiased under standard Gauss-Markov assumptions but suffers from high variance when features are correlated or when the feature dimension $p$ approaches the sample size $N$. In insurance feature sets with dozens of correlated features — multiple premium aggregates, overlapping tenure metrics, collinear coverage amounts — OLS coefficient estimates become unstable. Small perturbations in the data produce large swings in the coefficient vector.

### Ridge Regression ($\ell_2$ Penalty)

Ridge regression introduces a quadratic penalty on the coefficient magnitudes:

$$
\hat{\boldsymbol{\beta}}_{\text{Ridge}} = \arg\min_{\boldsymbol{\beta}} \sum_{i=1}^{N} (y_i - \mathbf{x}_i^\top \boldsymbol{\beta})^2 + \lambda \|\boldsymbol{\beta}\|_2^2
$$

This trades a controlled amount of bias for a substantial reduction in variance. The $\ell_2$ penalty handles multicollinearity gracefully: rather than arbitrarily assigning weight to one of several correlated features (as OLS does), Ridge distributes weight across the correlated group proportionally.

### Lasso Regression ($\ell_1$ Penalty)

Lasso performs simultaneous shrinkage and variable selection:

$$
\hat{\boldsymbol{\beta}}_{\text{Lasso}} = \arg\min_{\boldsymbol{\beta}} \sum_{i=1}^{N} (y_i - \mathbf{x}_i^\top \boldsymbol{\beta})^2 + \lambda \|\boldsymbol{\beta}\|_1
$$

The $\ell_1$ geometry induces exact zeros in the coefficient vector. This is valuable when many features are truly irrelevant, but it handles correlations poorly: among a group of correlated features, Lasso arbitrarily selects one and zeros the rest, which is unstable and difficult to interpret.

### Elastic Net (Combined $\ell_1 + \ell_2$)

Elastic Net combines both penalties:

$$
\hat{\boldsymbol{\beta}}_{\text{EN}} = \arg\min_{\boldsymbol{\beta}} \sum_{i=1}^{N} (y_i - \mathbf{x}_i^\top \boldsymbol{\beta})^2 + \lambda \Big[\alpha \|\boldsymbol{\beta}\|_1 + (1 - \alpha) \|\boldsymbol{\beta}\|_2^2 \Big]
$$

where $\alpha \in [0, 1]$ controls the $\ell_1 / \ell_2$ mixing.

### Interactive: Regularization Penalty Geometry

The geometric difference between $\ell_1$ and $\ell_2$ penalties explains their distinct behaviors. The constraint region for Ridge is a circle ($\|\boldsymbol{\beta}\|_2^2 \leq t$); for Lasso, it is a diamond ($\|\boldsymbol{\beta}\|_1 \leq t$). The diamond's corners lie on the coordinate axes, making it likely that the loss function's contours intersect the constraint boundary at a vertex — producing an exact zero in the coefficient.

> [!info] Interactive Element
> This section contained an interactive visualization in the original post.

> [!info] Interactive Element
> This section contained an interactive visualization in the original post.

### The Choice for This Model

The reinstatement model's base logistic regression uses **pure Ridge** ($\alpha = 0$) with a moderate regularization strength. This reflects the feature structure: most features carry some predictive signal (so Lasso's variable elimination is undesirable), but multicollinearity among premium, tenure, and coverage features requires shrinkage for stable coefficient estimation. Ridge's distributed weight allocation across correlated features produces more interpretable and stable coefficients than Lasso's arbitrary selection.

## Handling Class Imbalance

With a single-digit positive rate, a model predicting "no reinstatement" for all observations achieves high accuracy and zero business value. Rather than synthetic oversampling (SMOTE) or random duplication, the model uses cost-sensitive instance weighting:

$$
w_i = \begin{cases}
n_- / n_+ & \text{if } y_i = 1 \\
1.0 & \text{if } y_i = 0
\end{cases}
$$

Each positive sample contributes proportionally more to the loss gradient, forcing the model to learn minority-class patterns without distorting the feature space. This preserves the real class distribution (important for calibrated probability estimates) while addressing the gradient imbalance. Both the logistic regression base and the XGBoost residual model receive these weights.

## Model Architecture

### Stage 1: Base Model (Logistic Regression)

$$
\mathcal{L}_{\text{base}}(\mathbf{w}, b) = -\frac{1}{N}\sum_{i=1}^{N} w_i \Big[ y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i) \Big] + \lambda \|\mathbf{w}\|_2^2
$$

where $\hat{p}_i = \sigma(\mathbf{w}^\top \mathbf{x}_i + b)$, with Ridge regularization and cost-sensitive weights $w_i$.

Pipeline: `StringIndexer` $\to$ `OneHotEncoder` $\to$ `VectorAssembler` $\to$ `StandardScaler` $\to$ `LogisticRegression`.

### Stage 2: Residual Model (XGBoost Regressor)

$$
r_i = y_i - \hat{p}_{\text{base},i}
$$

$$
\mathcal{L}_{\text{residual}}(\mathbf{\Theta}) = \frac{1}{N}\sum_{i=1}^{N} \Big(r_i - f_{\text{XGB}}(\mathbf{x}_i; \mathbf{\Theta})\Big)^2 + \alpha \sum_k |w_k| + \frac{1}{2}\lambda \sum_k w_k^2
$$

Standard moderate hyperparameters (shallow trees, conservative learning rate, column/row subsampling, combined $\ell_1 + \ell_2$ regularization). All fixed upfront — no grid search on the residual model.

### Ensemble Combination

$$
\hat{y}_{\text{final},i} = \text{clip}\!\Big(\hat{p}_{\text{base},i} + f_{\text{XGB}}(\mathbf{x}_i; \hat{\mathbf{\Theta}}),\; 0,\; 1\Big)
$$

This two-stage residual architecture is the same framework used in the [[portfolio/contact-model-two-stage-residual-learning/index|outbound contact prediction model]], where the logistic regression captures population-level linear effects and the gradient-boosted residual stage corrects for non-linear interactions.

## Validation

Time-based OOS split with a **maturity gap** between training and test to allow the target window to resolve fully. Training spans multiple months; testing uses one holdout month.

A further sub-split within the training set monitors overfitting. AUC gaps exceeding 0.05 between sub-train and validation trigger warnings.

### Data Leakage Detection

A lightweight random forest is fitted on a 20% sample as a diagnostic. Any single feature with importance $> 50\%$ triggers a leakage alert; $> 30\%$ triggers a caution flag. The reinstatement history feature dominates — flagged for review but confirmed as a legitimate behavioral signal.

### Decile Performance

Top-decile reinstatement rates substantially exceeded the population average. The top fraction of the ranked list captured the majority of actual reinstatements. Discrimination was sharp across deciles, with bottom-half deciles contributing few reinstatements.

### Priority Score

Customers with multiple lapsed policies represent higher revenue recovery per successful contact. A priority score combines model probability with policy count:

$$
\text{PriorityScore}_i = \hat{p}_i \times \text{TargetPolicyCount}_i
$$

This reranking improved customer-level targeting in the top decile by surfacing multi-policy customers with moderate individual reinstatement probabilities but high aggregate expected value.

### Calibration

Model calibration was evaluated by comparing expected vs. actual reinstatement counts across train, validation, and test splits. The model showed slight conservatism on the test set (mild underprediction), which is acceptable — the business can apply a multiplicative correction if exact volume forecasting is needed.

### Feature Interpretation

Top logistic regression coefficients:

1. **Prior reinstatement count** (positive): the single strongest predictor — customers who have reinstated before are substantially more likely to do so again
2. **Payment modality** (negative): certain payment methods correlate with lower reinstatement odds
3. **Contract tenure** (positive): longer-tenured customers have more invested in their policies (sunk cost effect)
4. **Missed payment indicators** (positive): counterintuitively, higher missed payment counts correlate with *higher* reinstatement — these customers are in temporary financial distress, not permanent disengagement
5. **Multi-policy indicator** (positive): customers with multiple policies have more to lose from lapsing

> [!note]
> ## Interpretation: Missed Payments as a Positive Signal

The missed payment finding is notable. It suggests the model distinguishes between temporary cash flow difficulty (customers who eventually recover and reinstate) and permanent disengagement (customers who never intended to continue paying).

## AI-Assisted Development

Gemini helped design the value scoring scheme and the priority score formulation. Claude Code generated four PySpark scripts from `.md` prompt specifications: base feature engineering, value score computation, missed payment and loan feature extraction, and the preprocessing pipeline — plus the training script with cost-sensitive two-stage ensemble, decile analysis, calibration checks, and leakage detection.

## Technical Stack

| Layer | Technology |
|-------|-----------|
| Data Platform | Databricks, Delta Lake |
| Feature Engineering | PySpark (dozens of features, multiple source tables, target encoding) |
| Model | Logistic Regression + XGBoost (Two-Stage Residual, cost-sensitive) |
| Regularization | Ridge ($\ell_2$) for base model; $\ell_1 + \ell_2$ for residual model |
| Imbalance Handling | Class-proportional instance weighting |
| Validation | Time-based holdout with maturity gap, leakage detection |
| AI Workflow | Gemini (value scoring design) + Claude Code (4 FE scripts + training pipeline) |
