import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="All the Possum-bilities!")

st.markdown(
"""
<h1>All the Possum-bilities</h1>

Regression and possums galore!

<h2>Introduction</h2>
"""
, unsafe_allow_html=True)

# Load your dataset here
# Replace the following with your actual dataset loading method
# Example dataset (replace with actual data)

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

        # Display the metrics
        if mse is not None:
            st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
            st.write(f"**R-squared (R²):** {r2:.4f}")
            st.write(f"**Adjusted R-squared:** {adj_r2:.4f}")

            st.write("**Model Coefficients:**")
            st.write(coef_df)

            st.write("**Sample of Actual vs Predicted Values:**")
            st.write(comparison_df.head())
    else:
        st.write("Please select at least one feature to see the results.")

# Display visualization in the right column
with right_col:
    if fig is not None:
        st.pyplot(fig)



"""SLR INTERACTIVE GRAPH SECTION"""
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
st.title("SLR: Interactive Pairplots with Best Fit Line and Metrics")

# Display the plots in a 5-column layout
cols = st.columns(3)

# Iterate through the numerical predictors and assign plots to each column
for idx, predictor in enumerate(numerical_predictors):
    with cols[idx % 3]:
        fig = create_hoverable_plot(possum_df_enc, predictor, 'totlngth')
        st.plotly_chart(fig)