import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

# Load the data
df = pd.read_csv(r'C:\Users\youssef azam\Downloads\Amazon Sale Report.csv')

# Preprocess the data
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['ship-postal-code'] = df['ship-postal-code'].astype('object')

# Encode categorical variables
label_encoders = {}
categorical_cols = ['Fulfilment', 'Sales Channel ', 'ship-service-level', 'Style', 'SKU', 'Category', 'Size', 'ASIN', 'Courier Status', 'ship-city', 'ship-state', 'ship-country', 'promotion-ids', 'B2B', 'fulfilled-by', 'currency']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target variable
X = df.drop(columns=['Order ID', 'Date', 'Status'])
y = df['Status']

# Encode the target variable 'Status'
le_status = LabelEncoder()
y = le_status.fit_transform(y)

# Impute missing values
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object', 'int32']).columns

imputer_num = SimpleImputer(strategy='mean')
X[numerical_cols] = imputer_num.fit_transform(X[numerical_cols])

imputer_cat = SimpleImputer(strategy='most_frequent')
X[categorical_cols] = imputer_cat.fit_transform(X[categorical_cols])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Streamlit Dashboard
st.title('Amazon Sales Report Dashboard')

# Display data
st.subheader('Data Overview')
st.write(df.head())

# Visualizations
st.subheader('Visualizations')

# Status Count
st.write('### Order Status Count')
status_count = df['Status'].value_counts()
st.bar_chart(status_count)

# Fulfilment Count
st.write('### Fulfilment Count')
fulfilment_count = df['Fulfilment'].value_counts()
fig, ax = plt.subplots()
fulfilment_count.plot(kind='pie', autopct='%1.1f%%', ax=ax)
st.pyplot(fig)

# Amount Distribution
st.write('### Amount Distribution')
fig, ax = plt.subplots()
sns.histplot(df['Amount'], bins=30, kde=True, ax=ax)
st.pyplot(fig)

# Sales Trends Over Time
st.write('### Sales Trends Over Time')
monthly_sales = df.groupby(['Year', 'Month'])['Amount'].sum().reset_index()
monthly_sales['Date'] = pd.to_datetime(monthly_sales[['Year', 'Month']].assign(DAY=1))
fig, ax = plt.subplots()
sns.lineplot(x='Date', y='Amount', data=monthly_sales, ax=ax)
st.pyplot(fig)

# Top 10 Selling Products
st.write('### Top 10 Selling Products')
top_products = df.groupby('SKU')['Amount'].sum().nlargest(10).reset_index()
fig, ax = plt.subplots()
sns.barplot(y='Amount', x='SKU', data=top_products, palette='viridis', ax=ax)
st.pyplot(fig)

# Top 10 Selling Categories
st.write('### Top 10 Selling Categories')
top_categories = df.groupby('Category')['Amount'].sum().nlargest(10).reset_index()
fig, ax = plt.subplots()
sns.barplot(x='Amount', y='Category', data=top_categories, palette='viridis', ax=ax)
st.pyplot(fig)

# Sales Distribution by State
st.write('### Top 10 Sales Distribution by State')
state_sales = df.groupby('ship-state')['Amount'].sum().nlargest(10).reset_index()
fig, ax = plt.subplots()
sns.barplot(y='Amount', x='ship-state', data=state_sales, palette='viridis', ax=ax)
st.pyplot(fig)

# Order Quantities Distribution
st.write('### Distribution of Order Quantities')
fig, ax = plt.subplots()
sns.histplot(df['Qty'], bins=30, kde=True, ax=ax)
st.pyplot(fig)

# Sales by Fulfilment Type
st.write('### Sales by Fulfilment Type')
fulfilment_sales = df.groupby('Fulfilment')['Amount'].sum().reset_index()
fig, ax = plt.subplots()
sns.barplot(y='Amount', x='Fulfilment', data=fulfilment_sales, palette='viridis', ax=ax)
st.pyplot(fig)

# Sales by Sales Channel
st.write('### Sales by Sales Channel')
sales_channel_sales = df.groupby('Sales Channel ')['Amount'].sum().reset_index()
fig, ax = plt.subplots()
sns.barplot(y='Amount', x='Sales Channel ', data=sales_channel_sales, palette='viridis', ax=ax)
st.pyplot(fig)

# Correlation heatmap
st.write('### Correlation Heatmap')
corr = df[['Qty', 'Amount', 'Day', 'Month', 'Year']].corr()
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=.5, ax=ax)
st.pyplot(fig)

# Scatter plot of sales amount vs quantity
st.write('### Sales Amount vs Quantity')
fig, ax = plt.subplots()
sns.scatterplot(x='Qty', y='Amount', data=df, hue='Category', palette='viridis', ax=ax)
st.pyplot(fig)

# Box plot of sales amount by category
st.write('### Sales Amount Distribution by Category')
fig, ax = plt.subplots()
sns.boxplot(x='Category', y='Amount', data=df, palette='viridis', ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Sales by courier status
st.write('### Sales by Courier Status')
courier_status_sales = df.groupby('Courier Status')['Amount'].sum().reset_index()
fig, ax = plt.subplots()
sns.barplot(y='Amount', x='Courier Status', data=courier_status_sales, palette='viridis', ax=ax)
st.pyplot(fig)

# Sales by country
st.write('### Sales Distribution by Country')
country_sales = df.groupby('ship-country')['Amount'].sum().reset_index()
fig, ax = plt.subplots()
sns.barplot(x='ship-country', y='Amount', data=country_sales, palette='viridis', ax=ax)
st.pyplot(fig)

# Model Results
st.subheader('Model Results')

# Logistic Regression
st.write('### Logistic Regression')
st.write("Accuracy:", accuracy_score(y_test, y_pred_logreg))
st.write("Classification Report:\n", classification_report(y_test, y_pred_logreg))

# Decision Tree
st.write('### Decision Tree')
st.write("Accuracy:", accuracy_score(y_test, y_pred_dt))
st.write("Classification Report:\n", classification_report(y_test, y_pred_dt))

# Random Forest
st.write('### Random Forest')
st.write("Accuracy:", accuracy_score(y_test, y_pred_rf))
st.write("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Recommendations and Insights
st.subheader('Recommendations and Insights')

# Recommendations
st.write('### Recommendations')
st.write("""
1. **Optimize Fulfillment Process**: Given the distribution of orders by fulfillment type, focus on improving the efficiency of the most used fulfillment channels to enhance customer satisfaction and reduce delivery times.
2. **Promote Top-Selling Products**: Leverage the data on top-selling products and categories to create targeted marketing campaigns and promotions.
3. **Seasonal Sales Strategies**: Utilize the monthly sales trends to plan seasonal sales strategies, stock up on high-demand products during peak months, and offer discounts during slower periods to boost sales.
4. **Expand Popular Categories**: Invest in expanding the inventory for the top-selling categories to cater to the high demand and increase revenue.
5. **Geographical Targeting**: Based on the top 10 sales distribution by state, focus marketing efforts on regions with high sales and explore potential growth opportunities in lower-performing areas.
""")

# Insights
st.write('### Insights')
st.write("""
1. **Sales Trends**: There are noticeable trends in sales amounts over time, indicating peak seasons and slower periods. Align marketing strategies with these trends to maximize sales.
2. **Order Quantities**: The distribution of order quantities can help in inventory planning and management to ensure optimal stock levels and reduce overstock or stockouts.
3. **Fulfillment Impact**: Different fulfillment types have varying impacts on sales amounts. Analyzing these impacts can help in making strategic decisions to optimize fulfillment operations.
4. **Sales Channel Performance**: Understanding the performance of different sales channels can guide investment decisions and resource allocation to the most profitable channels.
5. **Correlation Insights**: The correlation between various numerical features like quantity, amount, and date components provides deeper insights into sales patterns and can aid in predictive analytics.
""")

