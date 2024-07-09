# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 15:24:02 2024

@author: Salma Badr
"""


'''
2. song length v. popularity: check for nonlinear relationship
maybe sort into categories - less than 4 minutes  vs. more than 4 minutes

3. z-score before using parametric test


'''



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import statsmodels.formula.api as smf
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import random 
from matplotlib.colors import ListedColormap

random.seed = 14984401

def plotHistWNormal(data, bins=30, xlabel='Value', ylabel='Frequency'):
    plt.clf()
    mu, std = norm.fit(data)
    xmin, xmax = min(data), max(data)
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.hist(data, bins=bins, density=True, alpha=0.6, color='b', label= xlabel + ' Data')
    plt.plot(x, p, 'k', linewidth=2, label=f'Fit results: $\mu={mu:.2f}$, $\sigma={std:.2f}$')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

def mannWhitneyReport(sample1,sample2):
    u_test, p_val = stats.mannwhitneyu(sample1, sample2)
    print(f"U-VAL: {u_test}")
    print(f"p-val: {p_val:.4f}")
    if p_val < 0.005:
        print("P < 0.005")
        return True
    return False

def calcRSquared(dataFrame, predictors, outcome):
    focusedDF = dataFrame[[predictors,outcome]].dropna()
    x = focusedDF[[predictors]]
    y = focusedDF[outcome]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random.seed)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return r2

def calcRSquared2(dataFrame, predictors, outcome):
    elements = predictors + [outcome]
    focusedDF = dataFrame[elements].dropna()
    x = focusedDF[predictors]
    y = focusedDF[outcome]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random.seed)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return r2
    
def calcRMSE(dataFrame, predictors, outcome):
    focusedDF = dataFrame[[predictors,outcome]].dropna()
    x = focusedDF[[predictors]]
    y = focusedDF[outcome]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random.seed)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return np.sqrt(mse)

def logRegMode(dataFrame, predictor):
    modeDF = dataFrame[['mode', predictor]].dropna()
    x = modeDF[[predictor]]
    y = modeDF['mode']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random.seed)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    confMatrix = confusion_matrix(y_test, y_pred)
    totalPos = confMatrix[1][0] + confMatrix[1][1]
    totalNeg = confMatrix[0][0] + confMatrix[0][1]
    print(confMatrix[0][1])
    print(f"Confusion Matrix: \n{confMatrix}")
    print(y_test.sum())
    try: 
        truePosRate = confMatrix[1][1]/totalPos
    except:
        print("No major key songs found in test set")
    else:
        print("True Positive Rate:", truePosRate)
    try:
        trueNegRate = (confMatrix[0][0])/totalNeg
    except:
        print("No minor key songs found in test set")
    else: 
        print("True Negative Rate:", trueNegRate)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC Score: {auc}")
    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='#FF69B4', lw=2, label=f'ROC curve (area = {auc:.2f})')
    ax.plot([0, 1], [0, 1], color='#89CFF0', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Receiver Operating Characteristic (ROC) Curve of {str.title(predictor)} Logistic Regression Model')
    ax.legend(loc="lower right")
    plt.show()

def plotModelCorr(dataFrame, predictors, outcome):
    focusedDF = dataFrame[[predictors,outcome]].dropna()
    x = focusedDF[[predictors]]
    y = focusedDF[outcome]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random.seed)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    predRealDict = {'predictions':y_pred, 'actuals': y_test}
    predRealDF = pd.DataFrame(predRealDict)
    plot = predRealDF.plot.scatter(x='actuals',y='predictions',alpha =0.1, color = '#FF69B4')
    plot.set_title(f"Scatterplot of Predicted and Actual {str.title(outcome)} Values of {str.title(predictors)} Regression Model (R^2 = {r2:.3f})")
    plot.set_xlabel(f"Actual {str.title(outcome)} Values")
    plot.set_ylabel(f"Predicted {str.title(outcome)} Values")
    plt.show()
   
def logRegMultiple(dataFrame, predictors, outcome, title = 'Receiver Operating'):
    modelElements = predictors + [outcome]
    df = dataFrame[modelElements].dropna()
    x = df[predictors]
    y = df[outcome]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random.seed)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    confMatrix = confusion_matrix(y_test, y_pred)
    totalPos = confMatrix[1][0] + confMatrix[1][1]
    totalNeg = confMatrix[0][0] + confMatrix[0][1]
    print(confMatrix[0][1])
    print(f"Confusion Matrix: \n{confMatrix}")
    print(y_test.sum())
    truePosRate = confMatrix[1][1]/totalPos
    print("True Positive Rate:", truePosRate)
    trueNegRate = (confMatrix[0][0])/totalNeg
    print("True Negative Rate:", trueNegRate)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC Score: {auc}")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='#FF69B4', lw=2, label=f'ROC curve (area = {auc:.2f})')
    ax.plot([0, 1], [0, 1], color='#89CFF0', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Receiver Operating Characteristic (ROC) Curve of {str.title(predictors[0])} Logistic Regression Model')
    ax.legend(loc="lower right")
    plt.show()


print("WE ARE WORKING WITH AN ALPHA THRESHOLD OF 0.005")
songs = pd.read_csv("spotify52kData.csv")
songFeatures = ['duration','danceability','energy','loudness','speechiness','acousticness','instrumentalness','liveness','tempo','valence']
plotHistWNormal(songs['popularity'], xlabel = "pop")


# [tn, fp]
# [fn, tp]


# for each feature, plot a histogram with a fit normal distribution for contrast
for column in songFeatures:
    plotHistWNormal(songs[column],bins = 30, xlabel = str.title(column), ylabel = 'Frequency')



# Q2: Is there a relationship between song length and popularity of a song? 
# If so, if the relationship positive or negative?

# plot a scatter plot of predicted vs. actual values of popularity by a duration linear regression model

durationPopularity = songs[['duration', 'popularity']].dropna()

durationPopSpearman =  songs['duration'].corr(songs['popularity'], method = 'spearman')

print(durationPopSpearman)
durationPopModel = smf.ols('popularity ~ duration',data = durationPopularity).fit()
print(durationPopModel.summary())
print(f"Confidence Interval: {durationPopModel.conf_int(alpha=0.005)}")






# Yes, there is a relationship but it only explains only 0.3% of the variance in popularity

# Q3: Are explicitly rated songs more popular than songs that are not explicit? [Suggestion: Do a suitable significance test, be it parametric, non-parametric or permutation]

# separate songs based on whether they are explicitly rated or not
# and perform mann whitney u test to determine if the difference in their median
# popularity is statistically significant
explicitPopDF = songs[['explicit','popularity']].dropna()
clean = explicitPopDF[explicitPopDF['explicit']==False]
explicit = explicitPopDF[explicitPopDF['explicit']==True]
mannWhitneyReport(clean['popularity'],explicit['popularity'])

# calculate respective median popularities
print(np.median(clean['popularity']))
print(np.median(explicit['popularity']))


# Q4: Are songs in major key more popular than songs in minor key? 

# separate songs into major key songs and minor key songs
# and perform mann whitney u test to determine if the difference in their median
# popularity is statistically significant
modePopDF = songs[['mode','popularity']].dropna()
major = modePopDF[modePopDF['mode']== 1]
minor = modePopDF[modePopDF['mode']== 0]
mannWhitneyReport(major['popularity'],minor['popularity'])
# calculate respective median popularities
print(np.median(major['popularity']))
print(np.median(minor['popularity']))

print("THE DIFFERENCE IN MEDIANS OF MAJOR AND MINOR KEY SONGS IS SIGNIFICANT. MINOR KEY SONGS ARE MORE POPULAR THAN MAJOR KEY SONGS.")



# Q5: Energy is believed to largely reflect the “loudness” of a song. 
# Can you substantiate (or refute) that this is the case? [Suggestion: Include a scatterplot]

# perform EDA: plot a scatterplot of energy and loudness and observe relationship
# I did this, then observed the relationship was nonlinear, but monotonic 
# then decided to add the spearman correlation coefficient to the figure
# to quantify the strength of the relationship
energyLoudnessDF = songs[['energy','loudness']].dropna()
energyLoudSpearmanCoeff =  energyLoudnessDF['energy'].corr(energyLoudnessDF['loudness'], method = 'spearman')
print(f"Spearman Correlation Between Energy and Loudness: {energyLoudSpearmanCoeff}")
plot = energyLoudnessDF.plot.scatter(x = 'energy',y= 'loudness',alpha = 0.05, color = '#9370DB')
plot.set_title(f"Scatterplot of Energy and Loudness (p = {energyLoudSpearmanCoeff:.2f})")





# Q6: Which of the 10 individual (single) song features fromquestion 1 predicts popularity best? How good is this “best” model?

# train a model to predict popularity from each of the song features, 
# then plot their pred v. real values and display their R^2 values
for column in songFeatures:
    plotModelCorr(songs, column, 'popularity')
    
print("Instrumentalness is the best predictor of popularity.")
print("A Linear Regression Model with Instrumentalness as a predictor explains 2% of the variance in popularity")
print(f"RMSE of model: {calcRMSE(songs,'instrumentalness','popularity')}")




# Q7: Building a model that uses *all* of the song features fromquestion 1, 
# how well can you predict popularity now? How much (if at all) is this model 
# improved compared to the best model in question 6). How do you account for 
# this?

# train a ridge regression model to predict popularity from all song features

modelElements = songFeatures + ['popularity']
songFeaturesDF = songs[modelElements].dropna()

x = songFeaturesDF[songFeatures]
y = songFeaturesDF['popularity']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random.seed)

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)


y_pred = ridge_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# plot the pred v. real values of the model and display the R^2
predRealDict = {'predictions':y_pred, 'actuals': y_test}
predRealDF = pd.DataFrame(predRealDict)
plot = predRealDF.plot.scatter(x='actuals',y='predictions',alpha =0.1, color = '#FF69B4')
plot.set_xlabel("Actual Popularity Values")
plot.set_ylabel("Predicted Popularity Values")
plot.set_title(f"Scatterplot of Predicted and Actual Popularity Values of All Feature Ridge Regression Model (R^2 = {r2:.3f})")


print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R²):", r2)


# It can explain 3% more of the variance than the instrumentalness model

# Q8: When considering the 10 song features above, how many meaningful principal 
# components can you extract? What proportion of the variance do these principal 
# components account for?

# step 1, create a correlation matrix of all of the features


# perform EDA to observe multi-collinearity
songsDF = songs[songFeatures]
corr_matrix = songsDF.corr()

with pd.option_context('display.max_columns', None):
    with pd.option_context('display.max_rows', None):
        print(corr_matrix)



# PCA


scaler = StandardScaler()
songsScaled = scaler.fit_transform(songs[songFeatures])

pca = PCA()
pca.fit(songsScaled)
totalExplainedVar = 0
screeDict = {}
# iterate through the eigenvalues of each principal components
# once we reach the first one that is less than 1, stop and note the number of components
for i, explainedVar in enumerate(pca.explained_variance_):
    totalExplainedVar += explainedVar
    screeDict[i+1] = explainedVar
    if explainedVar < 1 and screeDict[i] > 1:
        numComponents = i
        print(f"Using the Kaiser Criterion, we can extract {numComponents} principal components. \nThey account for {totalExplainedVar:.2f}% of the variance in the dataset")

# plot the eigenvalues and a horizontal line corresponding to the Kaiser criterion
screeDict2 = {"principalComponents": screeDict.keys(), 'eigenvalues': screeDict.values()}
screeDF = pd.DataFrame(screeDict2)
plot = screeDF.plot.bar(x= 'principalComponents',y= 'eigenvalues', color = '#9370DB')
plot.set_xlabel("Principal Components")
plot.set_ylabel("Eigenvalues")
plot.set_title("PCA Screeplot")
plt.axhline(y=1, color='#FF69B4', linestyle='--', linewidth=2)
plt.show()


# look at how each principal component is correlated with the songFeatures data
'''
for i in range(numComponents):
    componentCorrs = []
    for n in range(len(pca.components_[i])):
        componentCorrs += [pca.components_[i][n]]
    featureCompCorrs = dict(zip(songFeatures,componentCorrs))
    print(f"PC{i+1}:")
    print()
    '''
correlation_data = []
for i in range(numComponents):
    componentCorrs = []
    for n in range(len(pca.components_[i])):
        componentCorrs.append(pca.components_[i][n])
    featureCompCorrs = dict(zip(songFeatures, componentCorrs))
    print(f"PC{i+1}:")
    print()
    roundedCorrs = {k: round(v, 2) for k, v in featureCompCorrs.items()}
    correlation_data.append(roundedCorrs)
colors = ['#FFB6C1', '#FFDAB9', '#B0E0E6', '#98FB98', '#FFA07A', '#FFC0CB', '#E0FFFF', '#FFD700', '#AFEEEE', '#FF6347']
cmap = ListedColormap(colors)
corrdf = pd.DataFrame(correlation_data)
plot = corrdf.plot.bar(figsize=(10, 6), colormap=cmap)
plot.set_title('Principal Component Correlations')
plot.set_xlabel('Principal Components')
plot.set_ylabel('Correlation')
plt.xticks(range(3), range(1, 4), rotation=0)
plt.legend(prop={'size': 8})
plt.tight_layout()
plt.grid(True)
plt.show()

#Q9: Can you predict whether a song is in major or minor key from valence? If so, 
#how good is this prediction? If not, is there a better predictor? 
# [Suggestion: It might be nice to show the logistic regression once you are done building the model]


# create a logistic regression model for each feature and plot the ROC curve of each model
# and display AUC
for songFeature in songFeatures:
    print()
    print(f"Logistic Regression Using {songFeature} as a predictor: \n")
    logRegMode(songs,songFeature)


print()
print()


# No, you can't predict well from valence -- AUC = 50%
# speechiness is best reporting the highest AUC score of 57.02%


# Q10

# transform genre data to binary scale, 0 and 1, reflecting if song is classical

songs['track_genre'] = songs['track_genre'].apply(lambda x: 1 if x.lower() == 'classical' else 0)

# create logistic regression model predicting genre from duration and plot ROC
# and display AUC
logRegMultiple(songs,['duration'],'track_genre')


# create logistic regression model with principal components and plot ROC
# and display AUC
pca2 = PCA(n_components=numComponents)
X_pca = pca.fit_transform(songs[songFeatures])
y = songs['track_genre']
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
confMatrix = confusion_matrix(y_test, y_pred)
totalPos = confMatrix[1][0] + confMatrix[1][1]
totalNeg = confMatrix[0][0] + confMatrix[0][1]
print(confMatrix[0][1])
print(f"Confusion Matrix: \n{confMatrix}")
print(y_test.sum())

truePosRate = confMatrix[1][1]/totalPos
print("True Positive Rate:", truePosRate)

trueNegRate = (confMatrix[0][0])/totalNeg
print("True Negative Rate:", trueNegRate)
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC Score: {auc}")

fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='#FF69B4', lw=2, label=f'ROC curve (area = {auc:.2f})')
ax.plot([0, 1], [0, 1], color='#89CFF0', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC) Curve of Principle Components Logistic Regression Model')
ax.legend(loc="lower right")
plt.show()


# create a linear regression model predicting valence from loudness and report confidence interval
loudnessValenceDF = songs[['loudness','valence']].dropna()
results = smf.ols('valence ~ loudness', data = loudnessValenceDF).fit()

print(results.summary())
print(results.conf_int(alpha = 0.005))
# plot scatterplot of pred v. real values and display R^2
loudnessValenceCorr = loudnessValenceDF['loudness'].corr(loudnessValenceDF['valence'])
print(loudnessValenceCorr)
plotModelCorr(songs,'loudness','valence')
