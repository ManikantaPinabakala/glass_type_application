# Importing the necessary Python modules.
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Model building
# SVC model
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)

# Logistic Regression model
lg_clf = LogisticRegression(n_jobs = -1)
lg_clf.fit(X_train, y_train)

# Random Forest Classification model
rf_clf = RandomForestClassifier(n_jobs = -1, n_estimators = 100)
rf_clf.fit(X_train, y_train)

# Prediction function
@st.cache()
def prediction(model, RI, Na, Mg, Al, Si, K, Ca, Ba, Fe):
  glass_type = model.predict([[RI, Na, Mg, Al, Si, K, Ca, Ba, Fe]])
  label = glass_type[0]
  if label == 1:
    return "building windows float processed"
  elif glass_type == 2:
    return "building windows non float processed"
  elif glass_type == 3:
    return "vehicle windows float processed"
  elif glass_type == 4:
    return "vehicle windows non float processed"
  elif glass_type == 5:
    return "containers"
  elif glass_type == 6:
    return "tableware"
  else:
    return "headlamp"

#Titles
st.title("Glass Type Predictor")
st.sidebar.title("Exploratory Data Analysis")

#Displaying raw data
if st.sidebar.checkbox("Show raw data"):
  st.subheader("Full Dataset")
  st.dataframe(glass_df)

# Add a subheader in the sidebar with label "Scatter Plot".
st.sidebar.subheader("Scatterplot")
# Choosing x-axis values for the scatter plot.
# Add a multiselect in the sidebar with the 'Select the x-axis values:' label
# and pass all the 9 features as a tuple i.e. ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe') as options.
# Store the current value of this widget in the 'features_list' variable.
features_list = st.sidebar.multiselect("Select the X-axis value: ", ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))

st.set_option('deprecation.showPyplotGlobalUse', False)
for feature in features_list:
    st.subheader(f"Scatterplot between {feature} and glass_type")
    plt.figure(figsize = (15, 5))
    sns.scatterplot(x = feature, y = 'GlassType', data = glass_df)
    st.pyplot()

st.sidebar.subheader("Visualization sector")
plot_types = st.sidebar.multiselect("Select the plots:", ("Histogram", "Boxplot", "CountPlot", "Pie Chart", "Correlation Heatmap", "Pair Plot"))

if "Histogram" in plot_types:
  st.subheader("Histogram")
  columns = st.sidebar.selectbox("Select the column for histogram:", ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))

  plt.figure(figsize = (15, 5), dpi = 96)
  plt.title(f"Histogram for {columns}")
  plt.hist(glass_df[columns], bins = 'sturges', edgecolor = 'black')
  st.pyplot()

if "Boxplot" in plot_types:
  st.subheader("Boxplot")
  columns = st.sidebar.selectbox("Select the column for boxplot:", ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))

  plt.figure(figsize = (15, 5), dpi = 96)
  plt.title(f"Boxplot for {columns}")
  sns.boxplot(glass_df[columns])
  st.pyplot()

if 'CountPlot' in plot_types:
  st.subheader("Count plot")
  sns.countplot(x = 'GlassType', data = glass_df)
  st.pyplot()

if 'Pie Chart' in plot_types:
  st.subheader("Pie Chart")
  pie_data = glass_df['GlassType'].value_counts()

  plt.figure(figsize = (5, 5))
  plt.pie(pie_data, labels = pie_data.index, autopct = '%1.2f%%', startangle = 30, explode = np.linspace(.06, .16, 6))
  st.pyplot()

if 'Correlation Heatmap' in plot_types:
  st.subheader("Correlation Heatmap")

  plt.figure(figsize = (10, 6))
  # Creating an object of seaborn axis and storing it in 'ax' variable
  ax = sns.heatmap(glass_df.corr(), annot = True)
  # Getting the top and bottom margin limits.
  bottom, top = ax.get_ylim()
  # Increasing the bottom and decreasing the top margins respectively.
  ax.set_ylim(bottom + 0.5, top - 0.5)
  st.pyplot()

if 'Pair Plot' in plot_types:
  st.subheader("Pair Plots")

  plt.figure(figsize = (15, 15))
  sns.pairplot(glass_df)
  st.pyplot()

st.sidebar.subheader("Select your values:")

ri = st.sidebar.slider("Input Ri", float(glass_df['RI'].min()), float(glass_df['RI'].max()))
na = st.sidebar.slider("Sodium", float(glass_df["Na"].min()), float(glass_df["Na"].max()))
mg = st.sidebar.slider("Magnesium", float(glass_df["Mg"].min()), float(glass_df["Mg"].max()))
al = st.sidebar.slider("Aluminium", float(glass_df["Al"].min()), float(glass_df["Al"].max()))
si = st.sidebar.slider("Silicon", float(glass_df["Si"].min()), float(glass_df["Si"].max()))
k = st.sidebar.slider("Potassium", float(glass_df["K"].min()), float(glass_df["K"].max()))
ca = st.sidebar.slider("Calcium", float(glass_df["Ca"].min()), float(glass_df["Ca"].max()))
ba = st.sidebar.slider("Barium", float(glass_df["Ba"].min()), float(glass_df["Ba"].max()))
fe = st.sidebar.slider("Iron", float(glass_df["Fe"].min()), float(glass_df["Fe"].max()))

st.sidebar.subheader('Choose classifier')
classifier = st.sidebar.selectbox('Classifier', ('Support Vector Machine', 'Random Forest Classifier', 'Logistic Regression'))

if classifier == 'Support Vector Machine':
  st.sidebar.subheader('Model-parameters')
  c_values = st.sidebar.number_input('C[Error rate]', 1, 100, step = 1)
  kernel_input = st.sidebar.radio('Kernel', ('linear', 'rbf', 'poly'))
  gamma_input = st.sidebar.number_input('Gamma', 1, 100, step = 1)

  if st.sidebar.button('Classify'):
    st.subheader('Support Vector Machine')
    svc_model = SVC(C = c_values, kernel = kernel_input, gamma = gamma_input)
    svc_model.fit(X_train, y_train)
    y_predict = svc_model.predict(X_test)
    accuracy = svc_model.score(X_test, y_test)
    glass_type = prediction(svc_model, ri, na, mg, al, si, k, ca, ba, fe)
    st.write('The type of glass predicted is:', glass_type)
    st.write('Accuracy of the model:', accuracy.round(2))

    plot_confusion_matrix(svc_model, X_test, y_test)
    st.pyplot()

if classifier == 'Random Forest Classifier':
  st.sidebar.subheader('Model-parameters')
  n_estimators = st.sidebar.number_input('Number of trees in the forest', 100, 5000, step = 10)
  max_depth_input = st.sidebar.number_input('Maximum depth of a tree', 1, 100, step = 1)

  if st.sidebar.button('Classify'):
    st.subheader('Random Forest Classifier')
    rfc_model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth_input, n_jobs = -1)
    rfc_model.fit(X_train, y_train)
    accuracy = rfc_model.score(X_test, y_test)
    glass_type = prediction(rfc_model, ri, na, mg, al, si, k, ca, ba, fe)
    st.write('The type of glass predicted is:', glass_type)
    st.write('Accuracy of the model:', accuracy.round(2))

    plot_confusion_matrix(rfc_model, X_test, y_test)
    st.pyplot()

if classifier == 'Logistic Regression':
  st.sidebar.subheader('Model-parameters')
  c_values = st.sidebar.number_input('C', 1, 100, step = 1)
  max_iterations = st.sidebar.number_input('Maximum iterations', 10, 1000, step = 10)

  if st.sidebar.button('Classify'):
    st.subheader('Logistic Regression')
    log_reg = LogisticRegression(C = c_values, max_iter = max_iterations)
    log_reg.fit(X_train, y_train)
    accuracy = log_reg.score(X_test, y_test)
    glass_type = prediction(log_reg, ri, na, mg, al, si, k, ca, ba, fe)
    st.write('The type of glass predicted is:', glass_type)
    st.write('Accuracy of the model:', accuracy.round(2))

    plot_confusion_matrix(log_reg, X_test, y_test)
    st.pyplot()