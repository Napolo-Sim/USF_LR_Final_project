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

st.markdown(
    """
    <h1> All the Possum-bilities: The Mystery of Regression Methods! </h1>
    <div style="display: flex; align-items: center;">
        <div style="flex: 1; text-align: left;">
            <p>Hello I'm Sherlock Possulmes! Today we're going to be diving into the mystery of regression, and how we can use different types of models to predict information!</p>
            <p>I'm interested in how I can predict the size of my fellow possums based on their features. Can you help me solve this case?</p>
        </div>
        <div style="flex: 1; text-align: right;">
            <img src="https://cdn.discordapp.com/attachments/1270264700600586271/1294879989719765002/Untitled_design.jpeg?ex=670c9e77&is=670b4cf7&hm=c11dad745e08ba2af6ae9023fafc1194e1de1ebd24dd6e373b4c7241292e0039&" width="300">
        </div>
    </div>
    """, unsafe_allow_html=True
)


st.markdown(
"""
<h2>Introduction</h2>

As humans, you often use past experiences to make decisions or predictions. Maybe you've tried to guess how much a meal will cost based on the appearance of a restaurant, or estimated how tall a kid will grow based on their parent's heights. Whether you realize it or not, you are spotting patterns in past information and using them to make these predictions.

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
* Regression Tree

<p><em>As a bonus, we will also cover classification trees!</em></p>

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
* $\epsilon_i$: captures the random variation in $Y_i$ that cannot be explained by $X_i$


When we predict the dependent variable $Y_i$, we never know what the true coefficient $\\beta_1$ is since this value is calculated on the population level. We can only estimate it based on our sample, which gives us $\hat\\beta_1$. With our $\hat\\beta_1$, we can then find $\hat{Y_i}$ using this equation:
""", unsafe_allow_html=True)

st.latex(r'''
\hat{Y_i} = \hat{\beta_0} + \hat{\beta_1}X_i
''')

st.markdown(
"""
As a detective, we need to make sure our predictions are as spot on as possible, meaning we want the prediction or "best fit" line to be as close to the population's "best fit" line. To find the most accurate $\hat{Y_i}$ values, we consider criteria such as $R^2$ and $MSE$.

<h3>Measuring the Performance of the SLR Model</h3>

<h4>Mean-Squared Errors (MSE)</h4>

MSE measures the average squared error that occurs between actual and predicted values of Y. It is the average of the sum of all of the distances between our line (the prediction) and each of the actual data points, squared. The equation for MSE is:
""", unsafe_allow_html=True)

st.latex(r'''
MSE = \frac{1}{n-p} \sum\limits_{i=1}^n (Y_i-\hat{Y_i})^2
''')

st.markdown(
"""
Where
* $Y_i$ is the actual values of $Y_i$
* $\hat{Y_i}$ is the predicted values of $Y_i$
* $n$ is the number of observations
* $p$ is the number of $\\beta_j$'s (in SLR this value is 2)

The difference between the actual and predicted values of $Y_i$ is squared to keep these all of these values positive, and then summed and divided by $n-p$ to get MSE. Using MSE, we can figure out how well each individual factor predicts the actual values of our target variable. We want to **minimize MSE** AKA minimize the distance between our predictions and the actual points in order to find our best model.

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
In order words,  all the actual values minus the predicted values squared, summed up.

* $SST$ is the sum of squares total (the measure of the total variability in the dataset), and has the equation:
""", unsafe_allow_html=True)

st.latex(r'''
SST = \sum\limits_{i=0}^n (Y_i - \bar{Y_i})^2
''')

st.markdown(
"""
In other words, all the actual values minus the mean Y value squared, summed up.

$R^2$ can range between -1 and 1, but commonly falls between 0 and 1, where 0 indicates that the model does not explain any of the variance in the data, and a 1 indiciates the model explains all of the variance in the data. A negative $R^2$ value indicates that the model performs worse than just having a horizontal line at the mean of the data (our model is doing even worse than predicting the average!). **A higher $R^2$ close to 1 suggest the model is a good fit, while a lower $R^2$ suggests a poor fit.**

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

st.markdown(
"""
<h2>Multiple Linear Regression (MLR)</h2>

Multiple linear regression (MLR) further enhances our investigation into the mystery of possum length. Instead of just one clue now (one independent variable/predictor), we have multiple clues (two or more independent variables/predictors) to find the answer. The core of MLR still functions similarly to SLR, except our equation now needs to account for some more variables:
"""
, unsafe_allow_html=True)

st.latex(r'''Y_i = \beta_0+ \beta_1X_1 + \beta_2X_2+ ... + \beta_pX_{p-1} + \epsilon_i''')

st.markdown(
"""
We now have $X_1$ through $X_{p-1}$ where $p$ is the number of coefficients $\\beta$ we have in the model. These new predictors don't have to be uniform; we can combine categorical variables (which require preprocessing through dummy encoding, not covered here) with integer/float variables, enabling us to explore the relationships within the data more deeply!

<h3> Changes Compared to SLR </h3>

<h4> Multicollinearity and Variance Inflation Factors </h4>

While multiple predictors give us many more clues to work with, it also brings some new challenges to our investigation. One of the most signifcant challenges is multicollinearity, which can occur when two or more independent variables are highly correlated with each other. In other words, one variable explains the same effects as another variable, muddling the actual relationship between these variables and the dependent variable. High multicollinearity can affect our interpretation of the model and decrease the accuracy of our predictions.

To address multicollinearity in MLR, we can look at Variance Inflation Factors (VIF). VIF is a valuable tool that tells us whether an independent variable is correlated with another variable in the model.

Here's how we can interpret the values from VIF:
* **VIF between 1 and 4: low multicollinearity (preferred!)**
* **VIF between 4 and 10: moderate multicollinearity**
* **VIF above 10: high multicollinearity (needs to be addressed!)**

Calculating the VIF among the various independent variables helps us reduce multicollinearity by eliminating features with high VIF values, allowing us to draw more accurate conclusions from our data!

<h4> Adjusted R-squared </h4>

Another challenge is introduced with multiple predictor variables: as more independent variables are added to a model, $R^2$ will always increase even when the independent variables are not useful - or worse, are detrimental - to predicting the dependent variable. To solve this problem, we use adjusted $R^2$ written as $R_{adj}^2$.

$R_{adj}^2$ is similar to $R^2$ in that we are still measuring how well our model predicts the dependent variable, but also takes into account the effect that adding more predictors into the model has. An additional penalty is added for adding more variables to the model. The formula for $R_{adj}^2$ is:
"""
, unsafe_allow_html=True)

st.latex(r'''R^2_{adj} = 1 - \frac{MSE}{MST} = 1 - \frac{SSE/(n-p)}{SST/(n-1)}''')

st.markdown(
"""
Where
* $SSE$ is the sum of squared errors 
* $SST$ is the sum of squared total
(refer to equations above)
* $n$ is the total number of observations
* $p$ is the number of parameters ($\\beta_j$'s)

By using the alternative $R^2_{adj}$, we can better assess whether our new clues are genuinely improving our guess, or cluttering it with unnecessary complexity.

<h4>Advantages</h4>

MLR allows us to model more complex relationships between dependent and independent variables.
* MLR **accounts for more factors** that might usually influence the dependent variable.
* MLR **often gives more accurate and reliable predictions** since there are more factors to contribute to the prediction.
* MLR **can also model the relationship between two predictors (called an interaction)**, and how their presence together have an additional effect on the prediction.

<h4>Disadvantages</h4>

* Since MLR includes more factors, the model can become **harder to interpret**. It can be difficult to determine which factors have a small/large impact on the model.
* MLR is **more prone to overfitting** since we are considering more factors. There is a greater chance we are modelling the noise in the data as well.
* **Multicollinearity** is possible in MLR since we have more than 1 predictor, which can affect our interpretation of the model.

<h3> Building Intuition: Picking the Best MLR Model </h3>

To develop your intution on how the MLR model works in practice, we've prepared another interactive graph. In this graph, you can select which variables you would like to include in your model, and the graph plots the actual values (x) with the predicted values (y) from your chosen model. The $MSE$, $R^2$, and $R^2_{adj}$ are calculated and displayed under the graph, along with a table of some sample values.

Now that you've learned how the MLR model works and the new tool $R^2_{adj}$, let's put your investigative skills to the test again. What combination of variables increases $R^2$, but decreases $R^2_{adj}$?
"""
, unsafe_allow_html=True)

# Streamlit app
st.header("MLR: Find Features that Increase R Squared but Reduce R Squared Adjusted")

# Function to calculate adjusted R-squared
def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Function to train and display linear regression results
@st.cache_data
def train_and_display_linear_regression(selected_features):
    if not selected_features:
        return None, None, None, None, None, None

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    model = LinearRegression()
    model.fit(X_train_selected, y_train)

    y_pred_train = model.predict(X_train_selected)

    mse = mean_squared_error(y_train, y_pred_train)
    #rmse = np.sqrt(mse)
    r2 = r2_score(y_train, y_pred_train)

    n = len(y_train)
    p = len(selected_features)
    adj_r2 = adjusted_r2(r2, n, p)

    coef_df = pd.DataFrame({'Feature': selected_features, 'Coefficient': model.coef_})

    model = LinearRegression()
    model.fit(y_train.values.reshape(-1, 1), y_pred_train)  # Reshape for sklearn

    # Create a range of values for y_test for the regression line
    y_test_range = np.linspace(y_train.min(), y_train.max(), 100).reshape(-1, 1)
    y_pred_line = model.predict(y_test_range)

    # Create the scatter plot
    fig, ax = plt.subplots(figsize = (10, 8))
    ax.scatter(y_train, y_pred_train, color = '#fbada1')  # Adding some transparency for better visibility

    # Add a reference line (y = x)
    ax.plot(y_test_range, y_pred_line, color='#81776d', linestyle='-', lw=2, label='Regression Line')

    # Set labels and title
    ax.set_xlabel('Actual', fontsize=12, color='#81776d', weight='bold')
    ax.set_ylabel('Predicted', fontsize=12, color='#81776d', weight='bold')
    ax.set_title('Actual vs Predicted Values', fontsize=18, weight='bold', color='#81776d')

    # Hide the top and right spines and set spine colors and widths
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines['left'].set_color('#81776d')
    ax.spines['bottom'].set_color('#81776d')
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)

    # Change tick label colors
    ax.tick_params(axis='both', colors='#81776d')

    # Adjust label positions
    ax.xaxis.set_label_coords(0.85, -0.1)  # Adjust x label position
    ax.yaxis.set_label_coords(-0.1, 0.85)  # Adjust y label position

    # Show the plot
    plt.show()

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
        default=['age']  # Start with no features selected
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


st.markdown(
"""
There are many right answers for this question! One answer is a MLR model using the predictors hdlngth and age. The $R^2$ is 0.4948 and the $R^2_{adj}$ is 0.4885 for a model with only hdlngth. After adding the predictor age, our $R^2$ becomes 0.4959 (+0.0011) and the $R^2_{adj}$ becomes 0.4833 (-0.0052). If we were only looking at $R^2$, we might conclude that the second model is better, but looking at $R^2_{adj}$, we can conclude that the model with only hdlngth is better. $R^2_{adj}$ is essential in determining whether we are adding useful complexity or not.

Imagine you're looking at a more complex case, where the relationship between your clues and possum length isn't as straightforward. We can take a look at Classification and Regression Trees (CART) to uncover the patterns in our data.
"""
, unsafe_allow_html=True)


st.markdown(
"""
<h2> Regression Tree Models </h2>

Have you ever played 20 questions? A regression tree functions in almost a similar format, but without the 'sometimes' answer! You start with a broad question, and start narrowing down the possibilites based on the answer. Each question splits the possibilites into two subsets until you arrive at a prediction, giving you a tree-like structure.

If our goal is to predict the total length of a possum, we  can use regression trees to predict this numerical outcome. Is the possum older than 5? Is the possum's head length longer than 800mm? These are some examples of questions that can help us narrow down our prediction!

<!-- Regression trees also try to minimize the variance in the target variable using $MSE$. -->

<!-- The classification portion of CART refers to its ability to predict categorical outcomes (class labels like "spam" or "not spam").  -->

<!-- In other words, both classification and regression trees can use a combination of categorical and numerical input features, but the distinction lies in whether the target variable is categorical (for classification) or continuous (for regression). -->

Regression tree models have a couple base assumptions:

* The data is independent and identically distributed, meaning that all points are independent of one another.
    * In relation to our possum dataset, we assume that the data is collected in similar conditions and each measurement does not effect other ones.
* There is enough data in the training set to make meaningful splits and there is enough variety of data to find good splitting points.

One of the main differences from models, such as linear regression, is that regression tree models do not assume that the data follows any specific distribution.

<h4>Advantages</h4>

* Can predict **non-parametric relationships**, or relationships where the data does not follow a normal distribution.
* **Implicitly performs feature selection** by selecting the most important features at each split.
* **Outliers and missing values have minimal impact on its performance**, as the model relies on thresholds rather than assumed data distributions.
* Regression trees requires minimal supervision and **produces interpretable models**, with decision-making logic that is easy for most people to understand.

<h4>Disadvantages</h4>

* Regression tree models **tend to overfit the training data**, causing **high variance and low bias**, meaning that small changes in the data can lead to significantly different tree structures and heavily underperforming on new unseen data.
    * These issues can be addressed through methods such as pruning (think of this like chopping parts of a tree that are overgrown) to prevent the tree from growing too complex. 
    * The instability in performance can be mitigated using several decision trees and aggregating their results together (known as random forest) 
* Regresson trees also **rely on a greedy algorithm for splitting**, which selects the best local split at each step, even if it is not globally optimal for the overall model.

<h3> Base Model  </h3>


A basic regression tree contains decision nodes, branches, and their leafs which contain the prediction or classification results. We can compare these to elements of the 20 questions game:

* <b>Decision node</b>: Decision nodes are where you split the data, based on a specific feature. It asks a yes/no question, and sends the data down the left or right branch according to the answer.
    * Think of this as the question you get asked.
* <b>Branch</b>: Branches connect a decision node to the next decision node or tree leaf.
    * Based on a yes or no answer, move to another question!
* <b>Leaf</b>: Leafs are the end points of the tree, containing the different predictions.
    * This is the final answer, similar to the guess a 20 questions game gives you at the end!

In the diagram below, we have a short example of a decision tree. If the age is greater than 5 (Yes), then the predicted total length of the possum is 900mm.
"""
, unsafe_allow_html=True)

st.image("https://cdn.discordapp.com/attachments/1270264700600586271/1294825628536930376/Blank_diagram_-_Page_1.jpeg?ex=670c6bd6&is=670b1a56&hm=fe6e20b65bbfd175083143c4cf5a30df52abf215232c81cf026b85d0e1deb715&")


st.markdown(
"""
<h4> Building a Regression Model </h4>

Splitting is an essential part of regression trees to divide data into subsets we can make predictions on. Splitting lets us narrow down what the prediction is. In general, a regression model will stop splitting when:

* **All samples in a node are the same**: There’s no need to split further because all data points belong to one category or are identical.
* **No more features are left to split on**: All available features have been exhausted.
* **Only one data point remains in the node**: The model can't split a single data point.
* **Further splits don’t significantly improve predictions**: Splitting doesn’t lead to better accuracy or lower errors.

As a result of these factors, regression models are prone to overfitting. Overfitting occurs when a model learns the noise in the data as well as the patterns in the data. To minimize overfitting, we fine-tune our model using hyperparameters to prevent the model from getting too specific.

For our model, we chose to adjust three of the most common parameters: max depth, min samples split, and min samples leaf. However, there are several more parameters available that we won't be covering in this blog. These hyperparameters help control the complexity of the tree and improve its generalization to new data.

* **max_depth**: This controls how many tiers a tree can have below the initial decision node. In our visualization, this value would be 2. This limit helps prevent the tree from becoming too complex and overfitting the training data.
* **min_samples_split**: This defines the minimum number of data points needed in a group to allow for further splitting. Once a group reaches this number, it won’t split again, and the model will make a prediction at that leaf. This keeps the model simpler.
* **min_samples_leaf**: This sets the minimum number of data points required to create a leaf node. By increasing this number, the model becomes more reliable, ensuring that each final group formed by the tree has enough data to provide a good prediction.

You may be wondering, what is the best number to pick for these parameters? You may think that we should test each hyperparameter individually and increment their values and see what is the lowest MSE that we can achieve by changing the hyperparameters on their own and then combine them to make our best model, but let's see what that does.

Below are plots of each of our hyperparameters and their respective MSE values for each value of the hyperparameter incremented by 1. Based on these visualizations, what do you think are the best numbers to pick for each of the hyperparameters?
"""
, unsafe_allow_html=True)

# """3 INDEPENDENT HYPERPARAMETERS ANALYSIS FOR DECISION TREE"""
# st.title("Interactive Decision Tree Regressor")
# st.write("""
# This app allows you to interactively train a **Decision Tree Regressor** by modifying key hyperparameters:
# - Max Depth
# - Min Samples Split
# - Min Samples Leaf
# """)

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

st.markdown(
"""
<!-- DO WE WANT TO INLCUDE THESE? WE DIDN'T USE THIS
* <b>max_features</b>: Sets the maximum number of features that are considered when splitting at a node. This can help reduce the complexity of your model.
* <b>criterion</b>: This hyperparameter is used to set what impurity measure is being used to judge when to split. Some examples can be, the Gini coefficient, entropy or Mean Squared Errors which is typicaly used when doing regression trees.
* <b>min_impurity_decrease</b>: Is set to specify the minimum impurity decrease required for a split to occur.
-->

You may have thought that by looking at the graphs that the best max_depth value was 3, min_samples_split was 6, and that the best min_sample_leaf value was 4, but this is not always the case. When optimizing decision trees (or any machine learning model) using hyperparameters, it’s common to encounter situations where the best individual hyperparameters do not yield the best overall performance when combined. Some reasons why this can happen include:
1. **Interaction Effects Between Hyperparameters:** Hyperparameters can interact in complex ways. For example, the optimal value for the max_depth of a tree may depend on the value of min_samples_split. When optimizing them individually, you might miss these interactions that would only become apparent when they are optimized together.
2. **Overfitting or Underfitting:** The combination of hyperparameters can lead to overfitting or underfitting. One hyperparameter may be optimal for a specific range of values of another hyperparameter. For instance, a deeper tree (max_depth) might perform better with a higher minimum sample split (min_samples_split), but not with a lower one, and vice versa.
3. **Local Optima:** When tuning hyperparameters individually, you may be stuck in a local optimum. The best setting for one hyperparameter may not align well with the best setting of another. This is especially true in complex models where the performance landscape is not smooth.

To identify the optimal hyperparameters, we can use grid search—a systematic method for exploring various combinations of hyperparameter values. The goal is to compare all possible combinations to determine the best set (lowest MSE) for the model.

For our model, we tested the maximum depth of the tree with values ranging from 2 to 10, the minimum sample split from 2 to 10, and the minimum samples per leaf from 1 to 5. We  iterated through every combination of these hyperparameters, evaluating which combination produces the best regression tree based on the mean squared error (MSE).

<h3> Building Intuition: Picking the Best Hyperparameters for a Decision Tree </h3>
To build your intuition as to how the computer find the best hyperparameters to use, try finding the best combination of hyperparameters that minimize MSE. Does this match up with the best hyperparameters from the plots above?
"""
, unsafe_allow_html=True)

# """DECISION TREE INTERACTIVE GRAPH SECTION - START"""
# st.title("Interactive Decision Tree Regressor")
# st.write("""
# This app allows you to interactively train a **Decision Tree Regressor** by modifying key hyperparameters:
# - Max Depth
# - Min Samples Split
# - Min Samples Leaf
# """)

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
    # st.write(f"**R-squared (R²):** {r2:.4f}")
    # st.write(f"**Adjusted R-squared:** {adj_r2:.4f}")
    
    # Optionally display the comparison of actual vs predicted values
    # comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    # st.write(comparison_df.head())

# Call the function with current slider values
train_and_display_decision_tree(max_depth, min_samples_split, min_samples_leaf)


st.markdown(
"""
Based on your testing, were you able to identify the best hyperparameters for the model? You may have noticed that some individual hyperparameter values led to the lowest MSE, but also found that multiple min_samples_split values achieved the same result. This highlights the importance of optimizing hyperparameters as a combination rather than individually, as certain combinations can yield better performance than focusing on each hyperparameter in isolation.

Selecting the appropriate hyperparameters for our model is important, but it is also vital to understand how the decision tree determines where to split the data points. For this, regression trees use MSE, the same metric we discussed in the linear regression section, to guide their decisions on features and values to split.

<!-- <h4> Mean Squared Errors </h4>

Like other regression models, CART also uses Mean Squared Errors or $MSE$ to find the average squared difference between the actual and predicted values. Lower $MSEs$ indicate better splitting points. 

st.latex(r'''\begin{align}
MSE = \frac{1}{n}\sum\limits_{i=1}^{n}(y_i-\bar{y})^2
\end{align}''') -->

<h4> How are appropriate splits decided? </h4>

For regression trees, the model examines every possible split that can be formed based on the features and their values. This is similar to how 20 questions decided which questions are the best to ask earlier or later in the game.

**Numerical Features:** The algorithm evaluates all potential split points within a numerical feature, such as the weight of possums. For example, if the weights of the possums range from 1 kg to 10 kg, the algorithm might consider various split points, such as:

* Split at 3 kg: Group 1 (1 kg to 3 kg) vs. Group 2 (more than 3 kg)
* Split at 5 kg: Group 1 (1 kg to 5 kg) vs. Group 2 (more than 5 kg)
* Split at 7 kg: Group 1 (1 kg to 7 kg) vs. Group 2 (more than 7 kg)

**Categorical Features:** The algorithm creates binary splits based on unique categories. For instance, if you have a feature "Possum Primary Fur Color" with categories such as "Grey", "Beige", and "Black", the splits could be:

* "Grey" vs. "Beige and Black"
* "Beige" vs. "Grey and Black"
* "Black" vs. "Grey and Beige"

The algorithm then calculates the MSE for each of these splits and selects the one that results in the lowest impurity.

Each potential split is evaluated to find the one with the lowest impurity score. This process of finding the optimal split continues recursively to construct each decision node until one of the stopping criteria we mentioned above is met, like maximum tree depth. Now, we've finished building our regression tree model!

<!-- We can evaluate how well the model performs on a test set using MSE. -->


<h2> Classification Trees </h2>

Similar to regression trees, classification trees can be used to make predictions by asking questions in order to make a good guess. 
The main difference between regression and classification is that regression is trying to predict a continuous value whereas classification is 
trying to predict some type of category. Let's solidify your understanding with an intuition example.
""", unsafe_allow_html=True)

st.markdown(
"""
<h2>Building Intuition: How Do Classifiers Work?</h2>    

Let's start with a simple game of "Is it a possum or not?"

Given an animal, how could you guess if that given animal was a possum or not through some basic questions? Answer the questions below and see what the output is. 
"""
, unsafe_allow_html=True)

# """DECISION TREE - HOW IT WORKS VISUAL DEMO"""
def possum_decision_tree():
    # st.title("Possum Classification Decision Tree")

    st.write("Answer the following questions to classify if an animal is a possum:")

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
                st.write("**Classification: It is a Possum!**")
            else:
                st.write("**Classification: It is NOT a Possum!**")
        else:
            st.write("**Classification: It is NOT a Possum!**")

# Run the app
if __name__ == "__main__":
    possum_decision_tree()

st.image("https://cdn.discordapp.com/attachments/1270264700600586271/1294825902542422016/Blank_diagram_-_Page_1_1.jpeg?ex=670c6c17&is=670b1a97&hm=17dfd75772f78f0975170ac8744c9ce794991b84d2f6a4c8479c9b11fa0a715f&")

st.markdown(
"""
With this example, we can start getting the basic understandings of how classification trees work and begin to understand the difference between this method and regression. It functions in a similar way as regression trees except the way it splits points is based on impurity functions. The common impurity functions are Gini Index or Entropy and Information Gain.
<h3> New Impurity Functions </h3>

<h4> Gini Index/Impurity </h4>

The Gini Index, or Gini Impurity, helps evaluate potential split points in a classification tree. It estimates how likely it is to misclassify a randomly chosen item. Think of it as asking, "If I pick something at random, how often would I get its label wrong?" The goal is to create splits that group similar items together, reducing the chance of mistakes. A lower Gini Index means the split did a better job of organizing the data, making it easier for the model to make accurate predictions.
"""
, unsafe_allow_html=True)

st.latex(r'''
Gini(D) =1-\sum\limits_{i=0}^np_i^2
''')

st.markdown(
"""
<center>
<font size = "2">
Where p<sub>i</sub> is the proportion of instances belonging to class i in the subset D.
</font> 
</center>

<h4> Entropy </h4>

Entropy measures the uncertainty or disorder within a set of data. It helps determine the best split points in a classification tree by calculating how mixed or pure the data is. Think of it as asking, "How messy is this group?" A higher entropy value means the group is more diverse, with a mix of different classes. A lower entropy value means the group is more uniform, with most items belonging to the same class. The goal is to make splits that reduce entropy, organizing the data so the model can be more accurate. 
"""
, unsafe_allow_html=True)

st.latex(r'''
Entropy(D) = -\sum\limits_{i=1}^np_ilog_2(p_i)
''')

st.markdown(
"""
<center>
<font size="2"> 
With p<sub>i</sub> is the proportion of instances belonging to class i and subset D
</font>
</center> 

<h4> Information Gain </h4>

After calculating entropy, we can use this in conjunction with information gain. Information gain builds on to entropy by measuring how much entropy is reduced after a dataset is split. Think of it as asking, "How much clearer did things get after making this split?" A higher information gain means the split did a good job of organizing the data, creating child nodes that are more focused, with most items belonging to the same class. The goal is to make each split result in groups that are more uniform than before, helping the model make better predictions. Essentially, information gain tracks how much variability decreases with each split, similarly to MSE for regression.
"""
, unsafe_allow_html=True)

st.latex(r'''
Information Gain (D,A)= Entropy(D) - \sum\limits_{v\in Values(A)}\frac{|D_v|}{|D|}Entropy(D_v)
''')

st.markdown(
"""
<center>
<font size="2">  
Where D is the dataset, A is the atribute, D is the subset of D for which attribute A has the value, v and Values(A) are the posible values of attribute A. 
</font>
</center>


<h2>Model Selection</h2>

Now that we have many potential models to predict the total length of a possum, we need to narrow it down to the best model. The process of model selection is extremely important to ensure that we are choosing a model that most accurately predicts a possum length. We need to take into consideration factors such as the size of the dataset, number of predictors, model complexity, evaluation metrics, etc.

Using possums as an example, we find that the best model for the current data is the MLR model. To find the best model we can initially look at the $R_{adj}^2$ of all the models. Looking first at the regression tree we find that it is constantly predicting a negative $R_{adj}^2$ value indicating that the model does a poor job in predicting possum length (it does worse than simply predicting the average of all values). While the SLR models best $R^2$ is .48 it still falls short of MLR model which predicts a $R_{adj}^2$ of .78 indicating that it has the "best" predictive power of the three.

Another way to choose the best model is to compare the $MSE$ values of each model. When we compare $MSE$ for the three models, we find that the SLR model performs the worst, while the regression tree has a lower $MSE$ of 727 and $MLR$ has the best $MSE$ of 537. Therefore, the model that we should choose to solve my mystery is the MLR model.

Great job, possum pals! I hope you learned a lot today during our investigation into possum body lengths. We found the best model, and can now more accurately predict possum lengths. 

Signing off,
"""
, unsafe_allow_html=True) 

st.image("https://cdn.discordapp.com/attachments/1270264700600586271/1294851252043255879/signature.png?ex=670c83b3&is=670b3233&hm=8f0c4cc0bd460f3f1f3a6f6f5d8ca565be5fa16eba56800ab709b6e9c18f4cc9&")

st.markdown(
"""
Sherlock Possulmes
"""
, unsafe_allow_html=True)
