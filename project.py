import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import plotly.express as px

# Loading our dataset here w/ respective data transformations
possum_df = pd.read_csv('./Data/possum.csv')
possum_df = possum_df.drop(columns='site')
possum_df = possum_df.drop(columns='case')

# possum_df['site'] = possum_df['site'].astype("category")
possum_df['Pop'] = possum_df['Pop'].astype("category")
possum_df['sex'] = possum_df['sex'].astype("category")

possum_df_enc = pd.get_dummies(possum_df, columns=['Pop','sex'], drop_first=True)

# hdlngth in mm
# skullw in mm
# totallngth in cm => mm
# tail in cm => mm
# footlgth in => mm
# chest in cm => mm
# belly in cm => mm
possum_df_enc['totlngth'] = possum_df_enc['totlngth']*10
possum_df_enc['taill'] = possum_df_enc['taill']*10
possum_df_enc['chest'] = possum_df_enc['chest']*10
possum_df_enc['belly'] = possum_df_enc['belly']*10

# Fill missing values in column 'age' with the median of column 'age'
possum_df_enc['age'].fillna(possum_df_enc['age'].median(), inplace=True)
possum_df_enc['footlgth'].fillna(possum_df_enc['footlgth'].median(), inplace=True)

X = possum_df_enc.drop('totlngth', axis=1).copy()
y = possum_df_enc['totlngth']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

st.set_page_config(page_title="All the Possum-bilities!")

st.image("https://cdn.discordapp.com/attachments/1270264700600586271/1294825628536930376/Blank_diagram_-_Page_1.jpeg?ex=670c6bd6&is=670b1a56&hm=fe6e20b65bbfd175083143c4cf5a30df52abf215232c81cf026b85d0e1deb715&")

st.markdown(
"""
<h1> All the Possum-bilities </h1>

<!-- Hello I'm Sherlock Possulmes! -->

<h2>Introduction</h2>

As humans, we often use past experiences to make decisions or predictions. Maybe you've tried to guess how much a meal will cost based on the appearance of a restaurant, or estimated how tall a kid will grow based on their parent's heights. Whether you realize it or not, you are spotting patterns in past information and using them to make these predictions.

We can take that same idea and apply it towards predicting the length of a possum. What factors might you consider? Its weight? Its head length? Maybe even the size of its ears? These factors can help us make a more educated guess on the length of our furry friends. However, we can actually do more than an educated guess with the help of data and code.

Similar to humans, computers rely on existing data to make predictions through a process called regression. We'll take a dive into linear regression and look at how we can apply a model to predict the length of a possum. We'll also introduce an alternative regression technique called decision trees to show another way machines can make these predictions.

<h2>What is Regression?</h2>

Think of regression like being a detective - but instead of using clues to solve a crime, you're using data to uncover the relationships that help predict a specific outcome or variable. Consider how one thing (like possum length) can be solved by other clues (such as head length). Some clues are key to solving the mystery, while other clues will lead you on a wild goose chase.

In our investigation into possums, we're trying to predict how long a possum is. We have some potential suspects we might consider to solve the mystery, but it's up to us to figure out which ones really matter. Regression helps us narrow down the suspects to the ones that truly matter. We'll look at how strongly factors such as head length and ear conch length (our independent variables/predictors) affect the total length (our dependent variables/prediction).

Maybe we'll discover that all the independent variables demonstrate a strong linear relationship with our dependent variables or only a couple are needed to make the best prediction. Regression analysis lets us explore these complex relationships to discover which factors are pulling their weight, and which aren't. This way, we can make more accurate predictions efficiently, and uncover the mystery of possum lengths.

<h4>Models for Regression Analysis</h4>

We'll look at three common ways to perform a regression analysis:
* Simple Linear Regression
* Multiple Linear Regression
* Classification and Regression Trees (CART)

<h2>Simple Linear Regression</h2>

Let's start with the basics! Simple linear regression (SLR) is a great tool to draw a connection between a piece of evidence with what we are trying to discover, such as the independent variable head length and the dependent variable total length. Imagine you plot the data points on a coordinate grid, with the independent variables on the x-axis and the dependent variables on the y-axis. Your mission is to find the line that is the closest to all the points, or the line of best fit, in order to solve the mystery. 

More formally, the SLR model creates an equation for the line that best “fits” the data using this equation:
""", unsafe_allow_html=True)

st.latex(r'''
Y_i = \beta_0+ \beta_1X_i + \epsilon_i
''')

st.markdown("""
Where:
* $Y_i$: the dependent predicted variable (possum length in mm)
* $\\beta_0$: the first coefficient, aka the intercept
* $\\beta_1$: the coefficient of the independent variables
* $X_i$: the independent variables used to predict $Y_i$
* $\epsilon_i$: captures the random variation in $X_i$ that cannot be explained by $Y_i$


When we predict the dependent variable $Y_i$, we never know what the true coefficient $\\beta_1$ is since this value is calculated on the population level. We can only estimate it based on our sample, which gives us $\hat\\beta_1$. With our $\hat\\beta_1$, we can then find $\hat{Y_i}$ using this equation:
""", unsafe_allow_html=True)

st.latex(r'''
\hat{Y_i} = \hat{\beta_0} + \hat{\beta_1}X_i
''')

st.markdown(
"""
As a detective, we need to make sure our predictions are as spot on as possible, meaning we want the prediction or "best fit" line to be as close to the population's "best fit" line. To find the most accurate $\hat{Y_i}$ values, we consider criteria such as $R^2$ and $MSE$.

<h3>Measuring the Performance of the SLR Model</h3>

Now that we have a basic understanding of SLR, how do we know if our model or "best fit line" is good? Two key tools can help us measure the effectiveness of our model: $MSE$ and $R^2$.

<h4>Mean-Squared Errors (MSE)</h4>

MSE measures the average squared error that occurs between actual and predicted values of Y. It is the average of the sum of all of the distances between our line (the prediction) and each of the actual data points, squared. The equation for MSE is:
""", unsafe_allow_html=True)

st.latex(r'''
MSE = \frac{1}{n} \sum\limits_{i=0}^n (Y_i-\hat{Y_i})^2
''')

st.markdown(
"""
Where
* $Y_i$ is the actual values of $Y_i$
* $\hat{Y_i}$ is the predicted values of $Y_i$
* $n$ is the number of observations

The difference between the actual and predicted values of $Y_i$ is squared to keep these all of these values positive, and then summed and divided by $n$ to get MSE. Using MSE, we can figure out how well each individual factor predicts the actual values of our target variable. We want to **minimize MSE** AKA minimize the distance between our predictions and the actual points in order to find our best model.

<h4>R-squared (R<sup>2</sup>)</h4> 

Our other tool $R^2$ measures the amount of variance in a dependent variable that is explained by one or more of the independent variables. The equation for $R^2$ is:
""", unsafe_allow_html=True)

st.latex(r'''
R^2 = 1 - \frac{SSE}{SST}
''')

st.markdown(
"""
Where
* $SSE$ is the sum of squared errors and has the equation:
""", unsafe_allow_html=True)

st.latex(r'''
SSE = \sum\limits_{i=0}^n (Y_i - \hat{Y_i})^2
''')

st.markdown(
"""
* In order words,  all the actual values minus the predicted values squared, summed up.

* $SST$ is the sum of squares total (the measure of the total variability in the dataset), and has the equation:
""", unsafe_allow_html=True)

st.latex(r'''
SST = \sum\limits_{i=0}^n (Y_i - \bar{Y_i})^2
''')

st.markdown(
"""
* Or: all the actual values minus the mean Y value squared, summed up.

$R^2$ can range between -1 and 1, but commonly falls between 0 and 1. where 0 indicates that the model does not explain any of the variance in the data, and a 1 indiciates the model explains all of the variance in the data. A negative $R^2$ value indicates that the model performs worse than just having a horizontal line at the mean of the data (our model is doing even worse than predicting the average!). **A higher $R^2$ close to 1 suggest the model is a good fit, while a lower $R^2$ suggests a poor fit.**

<h3>Advantages & Disadvantages of SLR</h3>

<h4>Advantages</h4>

SLR is a basic form of regression analysis and can be powerful for many different tasks. Its advantages include:
* **Simplicity:** SLR is straightforward to model and interpret, involving just one predictor and one dependent variable.
* **Clear relationship:** SLR works well for models where there is a clear linear relationship between the dependent and independent variables.
* **Low Computational Cost:** SLR has a low computational cost compared to many other models due to its simplicity. SLR is quick to implement and can be run quickly on both small and large datasets.

<h4> Disadvantages </h4>

While SLR's simplicity makes it easy to run and implement, it is limited in how well it can capture more complex relationships. Its disadvantages include:
* **Assumption of linearity:** SLR assumes a linear relationship between the dependent and independent variables. If the data has a nonlinear relationship, SLR will not perform well.
* **Limited to one predictor:** many factors often contribute to the prediction of a dependent variable. In SLR, we can only consider the relationship between one independent variable and the dependent variable.
* **Sensitive to outliers:** SLR is highly sensitive to outliers. Outliers can heavily influence the slope of the line and cause the model to inaccurately predict new values.

<h2>Building Intuition: Picking the Best SLR Model</h2>

To develop your intution on how the SLR model works in practice, we've prepared an interactive graph that plots each independent variable versus the dependent variable (possum total length). As you hover over each of the plots, you will see values for $R^2$ and $MSE$ for each graph, uncover essential clues on which independent variables are the best predictor.

Now that you've learned how the SLR model works and the significance of $R^2$ and $MSE$, let's put your investigative skills to the test. Which independent variable do you believe is the best predictor of possum length? Take a moment to make your analysis, and check under the visualization to see if your detective hunches were correct!
""", unsafe_allow_html=True)

# """"""""""""""""""""""SLR INTERACTIVE GRAPH SECTION""""""""""""""""""""""
# Filter only numerical predictors
numerical_predictors = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Function to create a hoverable scatter plot with regression line and metrics
def create_hoverable_plot(df, predictor, response):
    X = df[[predictor]].values
    y = df[response].values

    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Compute metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Create scatter plot with trendline
    fig = px.scatter(df, x=predictor, y=response, trendline="ols", title=f'{predictor} vs {response}',
                     width=250, height=250)  # Adjust the size of the plots (square)
    
    fig.data[1].line.color = '#81776d' 
    fig.data[0].marker.color = '#fbada1'
    
    # Update hover template to include MSE and R² values
    fig.update_traces(
        hovertemplate=f'MSE: {mse:.2f}<br>R²: {r2:.2f}'
    )
    
    # Center the title and adjust margins
    fig.update_layout(
        title={
            'text': f'{predictor} vs {response}',
            'y': 0.95,  # Move the title closer to the top
            'x': 0.6,   # Center the title
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 12}
        },
        xaxis_title={'text': f'{predictor}', 'font': {'size': 10}},  # Adjust x-axis font size here
        yaxis_title={'text': f'{response}', 'font': {'size': 10}},
        margin=dict(l=10, r=10, t=30, b=10),  # Reduce left, right, top, bottom margins
    )

    return fig

# Streamlit app
# st.title("SLR: Interactive Pairplots with Best Fit Line and Metrics")

# Display the plots in a 5-column layout
cols = st.columns(3)

# Iterate through the numerical predictors and assign plots to each column
for idx, predictor in enumerate(numerical_predictors):
    with cols[idx % 3]:
        fig = create_hoverable_plot(possum_df_enc, predictor, 'totlngth')
        st.plotly_chart(fig)

st.markdown(
"""
If you chose **hdlength (head length)** as the best predictor of possum length, you're spot on! Head length had the lowest $MSE$ value and highest $R^2$ value, making head length the best variable to predict total possum length. Therefore, it will be the "best" model for this SLR case. 

However, keep in mind that the values of $MSE$ for this model were approximately 961, with an average $R^2$ value of 0.48. Could we lower $MSE$ or raise the $R^2$ value? To dive deeper into this mystery, lets turn to Multiple Linear Regression to see how we can refine our guesses.
"""
, unsafe_allow_html=True)




"""DECISION TREE INTERACTIVE GRAPH SECTION - START"""
st.title("Interactive Decision Tree Regressor")
st.write("""
This app allows you to interactively train a **Decision Tree Regressor** by modifying key hyperparameters:
- Max Depth
- Min Samples Split
- Min Samples Leaf
""")

# Step 7: Create sliders for parameter selection using Streamlit    

col1, col2, col3 = st.columns(3)  # Three columns for the sliders

# Create sliders in each column
with col1:
    max_depth = st.slider("Max Depth", min_value=2, max_value=10, value=5, step=1)

with col2:
    min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=10, value=2, step=1)

with col3:
    min_samples_leaf = st.slider("Min Samples Leaf", min_value=1, max_value=5, value=1, step=1)

# Define a function to calculate adjusted R-squared
def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Define a function to train the decision tree and display the tree
def train_and_display_decision_tree(max_depth, min_samples_split, min_samples_leaf):
    # Step 2: Define the Decision Tree Regressor with the chosen parameters
    regressor = DecisionTreeRegressor(random_state=42, 
                                      max_depth=max_depth, 
                                      min_samples_split=min_samples_split, 
                                      min_samples_leaf=min_samples_leaf)
    
    # Step 3: Fit the model with the training data
    regressor.fit(X_train, y_train)
    
    # Step 4: Plot the decision tree using plot_tree with default colors
    fig, ax = plt.subplots(figsize=(20, 10))
    
    plot_tree(regressor, feature_names=X_train.columns, filled=True, rounded=True, fontsize=12, ax=ax,
              node_ids=True, impurity=False, proportion=False, precision=2)

    plt.title(f"Decision Tree Regressor (max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf})",
              fontsize=18, fontname='Helvetica', color='#333333')

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Step 5: Make predictions on the test data
    y_pred = regressor.predict(X_test)
    
    # Step 6: Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate adjusted R-squared
    n = len(y_test)  # number of observations
    p = X_test.shape[1]  # number of predictors
    adj_r2 = adjusted_r2(r2, n, p)
    
    # Display performance metrics in Streamlit
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    # st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
    st.write(f"**R-squared (R²):** {r2:.4f}")
    st.write(f"**Adjusted R-squared:** {adj_r2:.4f}")
    
    # Optionally display the comparison of actual vs predicted values
    comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    st.write(comparison_df.head())

# Call the function with current slider values
train_and_display_decision_tree(max_depth, min_samples_split, min_samples_leaf)



"""3 SLIDERS SECTION FOR DECISION TREE"""
st.title("Interactive Decision Tree Regressor")
st.write("""
This app allows you to interactively train a **Decision Tree Regressor** by modifying key hyperparameters:
- Max Depth
- Min Samples Split
- Min Samples Leaf
""")

# Create three columns for plots
cols = st.columns(3)

# Define a function to calculate adjusted R-squared
def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Function to train the decision tree and calculate MSE for plotting
def calculate_mse(max_depth, min_samples_split, min_samples_leaf):
    # Define the Decision Tree Regressor
    regressor = DecisionTreeRegressor(
        random_state=42, 
        max_depth=max_depth, 
        min_samples_split=min_samples_split, 
        min_samples_leaf=min_samples_leaf
    )
    
    # Fit the model with the training data
    regressor.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = regressor.predict(X_test)
    
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    return mse

# Create sliders and plot MSE for each parameter
max_depth_values = range(2, 11)
min_samples_split_values = range(2, 11)
min_samples_leaf_values = range(1, 6)

# Plot for Max Depth
with cols[0]:
    mse_values = [calculate_mse(depth, 2, 1) for depth in max_depth_values]
    df_max_depth = pd.DataFrame({'Max Depth': max_depth_values, 'MSE': mse_values})
    
    fig_max_depth = px.line(df_max_depth, x='Max Depth', y='MSE', title='Max Depth vs MSE')
    fig_max_depth.update_traces(line=dict(color='#ffcdc4'))
    fig_max_depth.update_traces(mode='lines+markers', line=dict(color='#fbada1'))  
    st.plotly_chart(fig_max_depth)

# Plot for Min Samples Split
with cols[1]:
    mse_values = [calculate_mse(5, split, 1) for split in min_samples_split_values]
    df_min_samples_split = pd.DataFrame({'Min Samples Split': min_samples_split_values, 'MSE': mse_values})
    
    fig_min_samples_split = px.line(df_min_samples_split, x='Min Samples Split', y='MSE', title='Min Samples Split vs MSE')
    fig_min_samples_split.update_traces(line=dict(color='#ffcdc4'))
    fig_min_samples_split.update_traces(mode='lines+markers', line=dict(color='#fbada1'))  
    st.plotly_chart(fig_min_samples_split)

# Plot for Min Samples Leaf
with cols[2]:
    mse_values = [calculate_mse(5, 2, leaf) for leaf in min_samples_leaf_values]
    df_min_samples_leaf = pd.DataFrame({'Min Samples Leaf': min_samples_leaf_values, 'MSE': mse_values})
    
    fig_min_samples_leaf = px.line(df_min_samples_leaf, x='Min Samples Leaf', y='MSE', title='Min Samples Leaf vs MSE')
    fig_min_samples_leaf.update_traces(line=dict(color='#ffcdc4'))  
    fig_min_samples_leaf.update_traces(mode='lines+markers', line=dict(color='#fbada1'))  
    st.plotly_chart(fig_min_samples_leaf)


"""MULTIPLE LINEAR REGRESSION SECTION!!!"""
# Streamlit app
st.header("Multiple Linear Regression")
st.write("Select features to include in the model:")

# Function to calculate adjusted R-squared
def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Function to train and display linear regression results
@st.cache_data
def train_and_display_linear_regression(selected_features):
    if not selected_features:
        return None, None, None, None, None, None, None

    # Use training data for selected features
    X_train_selected = X_train[selected_features]
    
    # Train the model on the training data
    model = LinearRegression()
    model.fit(X_train_selected, y_train)

    # Predict on the training data
    y_pred_train = model.predict(X_train_selected)

    # Calculate metrics based on the training data
    mse = mean_squared_error(y_train, y_pred_train)
    r2 = r2_score(y_train, y_pred_train)

    n = len(y_train)  # Number of observations in the training set
    p = len(selected_features)  # Number of features
    adj_r2 = adjusted_r2(r2, n, p)

    coef_df = pd.DataFrame({'Feature': selected_features, 'Coefficient': model.coef_})

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(y_train, y_pred_train, color='#fbada1')

    # Add a reference line (y = x)
    ax.plot(y_train, y_train, color='#81776d', linestyle='-', lw=2, label='Regression Line')

    # Set labels and title
    ax.set_xlabel('Actual (Training)', fontsize=12, color='#81776d', weight='bold')
    ax.set_ylabel('Predicted (Training)', fontsize=12, color='#81776d', weight='bold')
    ax.set_title('Actual vs Predicted Values (Training Data)', fontsize=18, weight='bold', color='#81776d')

    # Customize spines and ticks
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines['left'].set_color('#81776d')
    ax.spines['bottom'].set_color('#81776d')
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.tick_params(axis='both', colors='#81776d')

    # Adjust label positions
    ax.xaxis.set_label_coords(0.85, -0.1)
    ax.yaxis.set_label_coords(-0.1, 0.85)

    # Show the plot
    plt.tight_layout()

    # Prepare comparison DataFrame for actual vs predicted values
    comparison_df = pd.DataFrame({'Actual': y_train, 'Predicted': y_pred_train})

    return mse, r2, adj_r2, coef_df, fig, comparison_df


# Create columns for dropdown and visualization
left_col, right_col = st.columns([1, 2])  # 1 part for dropdown and metrics, 2 parts for visualization

# Create dropdown in the left column
with left_col:
    feature_cols = X.columns.tolist()  # Get the feature columns
    selected_features = st.multiselect(
        "Choose features for regression:",
        options=feature_cols,
        default=[]  # Start with no features selected
    )

    # Initialize empty plot variable
    fig = None

    if selected_features:
        # Call the function to get MSE, R², etc. and plot
        mse, r2, adj_r2, coef_df, fig, comparison_df = train_and_display_linear_regression(selected_features)

        # Display the model coefficients
        st.write("**Model Coefficients:**")
        st.write(coef_df)

# Display visualization in the right column
with right_col:
    if fig is not None:
        st.pyplot(fig)

        # Display the metrics below the visualization
        if mse is not None:
            st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
            st.write(f"**R-squared (R²):** {r2:.4f}")
            st.write(f"**Adjusted R-squared:** {adj_r2:.4f}")

        # Display the sample of actual vs predicted values
        st.write("**Sample of Actual vs Predicted Values:**")
        st.write(comparison_df.head())
    else:
        if not selected_features:
            st.write("Please select at least one feature to see the results.")







"""DECISION TREE - HOW IT WORKS VISUAL DEMO"""
def possum_decision_tree():
    st.title("Possum Classification Decision Tree")

    st.write("Answer the following questions to classify if the animal is a possum:")

    # Question 1
    question1 = st.radio("Does the animal have a pointed snout?", ("Yes", "No"))

    if question1 == "Yes":
        # Question 2
        question2 = st.radio("Does it have a hairless tail?", ("Yes", "No"))

        if question2 == "Yes":
            st.write("**Classification: It is a Possum!**")
        else:
            st.write("**Classification: It is NOT a Possum!**")
    else:
        # Question 3
        question3 = st.radio("Is it nocturnal?", ("Yes", "No"))

        if question3 == "Yes":
            # Question 4
            question4 = st.radio("Does it have fur that can be gray, brown, or black?", ("Yes", "No"))

            if question4 == "Yes":
                st.write("**Classification: It is NOT a Possum!**")
            else:
                st.write("**Classification: It is NOT a Possum!**")
        else:
            st.write("**Classification: It is NOT a Possum!**")

# Run the app
if __name__ == "__main__":
    possum_decision_tree()
