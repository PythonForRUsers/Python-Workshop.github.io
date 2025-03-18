# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
import os

import pandas as pd
import numpy as np

from great_tables import GT

import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf

## import from sklearn (scikit-learn)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif

# Import display from IPython to allow display of plots in notebook
from IPython.display import display
#
#
#
#
#
#
#
#
#
data = pd.read_csv("quarto/session3/example_data/Cancer_Data.csv")
#
#
#
#
#
#
#
#
#
#
data.info()
#
#
#
#
#
#
#
#
## `inplace` means that we modify the original dataframe
data.drop(columns="Unnamed: 32", inplace=True)
data.columns = data.columns.str.replace(" ", "_")
## check that the column was removed
print(data.info())

#
#
#
#
#
#
#
data.head(5)
#
#
#
#
#
#
#
#
#
#
#
## define a dictionary
y_recode = {"B": 0, "M": 1}

## use .map() to locate the keys in the column and replace with values
## B becomes 0, M becomes 1
data["diagnosis"] = data["diagnosis"].map(y_recode)

data.head(5)
#
#
#
#
#
#
#
#
#
#
data.describe()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
summary = data.groupby("diagnosis").mean().reset_index()

(
    GT(summary)
    .tab_header(title = "SUMMARY")
    .tab_stub(rowname_col = "diagnosis")
    .tab_stubhead(label = "Diagnosis")
    .fmt_number(decimals = 2)
)

    
## ORRRRR

df = data[['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']]

df["size"] = pd.qcut(df["radius_mean"], q=3, labels = ["Small", 'Medium', 'Large'])

summary = df.groupby(['diagnosis', 'size']).agg(
    mean_radius_mean=("radius_mean", "mean"),  # Mean of the 'area' column
    mean_texture_mean=("texture_mean", "mean") , # Mean of the 'area' column,
    texture_mean = ("texture_mean", lambda x: " ".join(map(str, x))),  
)

summary = summary.reset_index()


(
    GT(summary)
    .tab_header(title = "SUMMARY")
    .tab_stub(rowname_col = "diagnosis", groupname_col = "size")
    .tab_stubhead(label = "Diagnosis")
    .fmt_number(decimals = 2)
)


#
#
#
#
#
#
#
#
#
#
#
#
#
#
from plotnine import ggplot, aes, geom_bar, labs
## OR
from plotnine import *
#
#
#
#
#
#
#
# | dpi: 600
## create a count plot
(
    ggplot(data, aes(x = 'diagnosis')) + 
    geom_bar() +
    labs(title = 'Distribution of Diagnoses', x = "Diagnosis", y = "Count")
)
#
#
#
#
#
#
#
#
#
sns.color_palette("colorblind")
#
#
#
#
#
#
#
#
# | message: False
color_hex = sns.color_palette("colorblind").as_hex()

print("The hexcodes for the 'colorblind' palette are:\n", color_hex)

## if we want to make the columns green for benign and yellow for malignant

## the "-" lets us index from the end of the list rather than the front.However, the '-1'th position is the last position (there is no '-0')

colors = {0: color_hex[2], 1: color_hex[-2]}
#
#
#
#
#
#
#
#
#
#
# Create count plot
(
    ggplot(data, aes(x='diagnosis', fill='factor(diagnosis)')) +
    geom_bar(aes(y='..count../sum(..count..)*100'), position='dodge') +
    scale_fill_manual(values=colors, labels=["Benign", "Malignant"]) +
    labs(title='Distribution of Diagnoses', x='Diagnosis', y='Percent', fill='Diagnosis') +
    theme_minimal()
)
#
#
#
#
#
#
#
#
#
#

correlation_matrix = (
    data.select_dtypes(include=[np.number])
    .drop(columns="id")  # Drop the 'id' column if present
    .corr()
)

corr_long = correlation_matrix.reset_index().melt(id_vars = 'index', 
var_name = 'Variable 1', value_name= "Correlation")

corr_long.rename(columns={'index': 'Variable 2'}, inplace=True)

# Create the correlation plot
plot = (
    ggplot(corr_long, aes(x='Variable 1', y='Variable 2', fill='Correlation')) +
    geom_tile() +
    scale_fill_gradient2(low='blue', mid='white', high='red', midpoint=0) +
    theme_minimal() +
    theme(axis_text_x=element_text(rotation=45, hjust=1)) +
    labs(title='Correlation Plot', x='Variable 1', y='Variable 2', fill='Correlation')
)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
logit = smf.logit("diagnosis ~ area_mean + texture_mean", data=data).fit()

print(logit.summary())
```
#
#
#
#
#
#
#
#
from sklearn.feature_selection import SelectKBest, f_classif

X_raw = data.loc[:, "radius_mean"::]
## set only the diagnosis column as "y"
y = data.loc[:, "diagnosis"]

# Select top k features based on ANOVA F-value between feature and target
selector = SelectKBest(f_classif, k=5)  # Choose 'k' to specify number of features
X_selected = selector.fit_transform(X_raw, y)
selected_feature_names = X_raw.columns[selector.get_support()]

## make model eqn
formula = "diagnosis ~" + "+".join(selected_feature_names)
sm_model = smf.logit(formula, data=data).fit()

print(sm_model.summary())
```
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
X = data.loc[:, "radius_mean"::]

## set only the diagnosis column as "y"
y = data.loc[:, "diagnosis"]

## here we assign each object returned from `train_test_split` to a different variable
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train.head(3)
#
#
#
#
#
#
#
## standardize dataset
scaler = StandardScaler()

## fit the scaler to the TRAINING data
scaler.fit(X_train)

## apply the scaler to BOTH the training and test data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#
#
#
#
#
#
#
## set up PCA transformer with the number of components you want and fit to training dataset
pca = PCA(n_components=10)
pca = pca.fit(X_train)

## apply PCA transformer to training and test set
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
#
#
#
#
#
#
#
#
#
#
#
#
#
## we can look at the cumulative explained variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
#
#
#
#
#
#
#
lr = LogisticRegression()
#
#
#
#
## fit to training data
lr.fit(X_train_pca, y_train)
#
#
#
#
#
#
#
#
#
#
#
## use model to predict test data
## set up dataframe to review results
results = pd.DataFrame()

## get predicted
results.loc[:, "Predicted"] = lr.predict(X_test_pca)

## get true y values for test dataset
results.loc[:, "Truth"] = y_test.values

## get probability of being malignant
## the output is one probability per outcome, we only want the second outcome (malignant). The second outcome uses index 1
results.loc[:, "Probability: Malignant"] = pd.DataFrame(lr.predict_proba(X_test_pca))[1]

results.head(5)
#
#
#
#
#
accuracy = accuracy_score(results["Truth"], results["Predicted"])

print("Accuracy: {:.2f}%".format(accuracy * 100))
#
#
#
#
#
#
#
#
#
#
#
# | fig-cap: An ROC curve for our logistic regression model

## make a plot to vizualize the ROC curve

## get false pos rate, true pos rate and thresholds
fpr, tpr, thresholds = roc_curve(results["Truth"], results["Predicted"])

## get AUC data
roc_auc = auc(fpr, tpr)

# Create a dataframe for plotting
roc_data = pd.DataFrame({"False Positive Rate": fpr, "True Positive Rate": tpr})

# Calculate accuracy for the title (using a threshold of 0.5)
predicted_class = (results["Predicted"] >= 0.5).astype(int)
accuracy = accuracy_score(results["Truth"], predicted_class)

# Plot using plotnine
roc_plot = (
    ggplot(roc_data, aes(x="False Positive Rate", y="True Positive Rate")) +
    geom_line(color="darkorange", size=1.5") +
    geom_abline(intercept=0, slope=1, linetype="dashed", color="navy", size=1) +
    labs(
        title=f"Receiver Operating Characteristic (ROC) Curve\nAccuracy: {accuracy * 100:.2f}%",
        x="False Positive Rate",
        y="True Positive Rate"
    ) +
    scale_color_manual(values={f"ROC Curve (AUC = {roc_auc:.2f})": "darkorange", "Random": "navy"})
    theme_minimal() +
    theme(
        legend_position="bottom",
        plot_title=element_text(size=14, face="bold"),
        axis_title=element_text(size=12)
    )
) )

print(roc_plot)

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
## scale X
X_raw = X  ## scaled dfs lose column names
X = scaler.transform(X_raw)

## set up model for Lasso and fit it
model = LogisticRegression(penalty="l1", solver="liblinear", C=0.01)
model.fit(X, y)

# Get non-zero coefficient features
selected_features = X_raw.columns[model.coef_[0] != 0]
X_selected = X_raw[selected_features]
print(X_selected.columns)
#
#
#
#
#
## Get coefficients
intercept = model.intercept_[0]
coefficients = model.coef_[0][model.coef_[0] != 0]

## make model eqn
formula = "diagnosis ~" + "+".join(X_selected.columns)
sm_model2 = smf.logit(formula, data=data).fit()

sm_model2.params[:] = np.concatenate(
    ([intercept], coefficients)
)  # Set params from scikit-learn model

# Display the summary
print(sm_model2.summary())
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
