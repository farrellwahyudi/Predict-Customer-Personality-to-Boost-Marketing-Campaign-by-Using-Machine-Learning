# Predict Customer Personality to Boost Marketing Campaign by Using Machine Learning
<img src="https://www.75seconds.com/wp-content/uploads/2021/06/Role-of-Animated-Explainer-Videos-for-Marketing-Success-of-Business.jpg" alt="marketing campaign" style="width:600px;height:400px;">

## Background
A company can experience rapid growth by gaining insights into the behavioral patterns of its customers, which in turn enables it to offer enhanced services and benefits to potential loyal customers. By utilizing historical marketing campaign data to enhance performance and effectively target customers for transactions on the company's platform, my primary objective is to create a predictive clustering model, streamlining decision-making for the company.

## Problem
1. The company does marketing campaigns for all of their customers. Doing this it has an average conversion rate per customer of ~ 0%.
2. The company spends a nonoptimal amount of marketing resources by campaigning to every single customer. It's marketing ROI is -73.08%.

## Objectives
1. Identify the factors that most influence customers' spending, purchases and conversion rate.
2. Create an optimal K-Means Clustering model that can decisively segment customers for marketing retargeting purposes.
3. Provide recommendations for potential strategies regarding targeted marketing based on findings from analyzes and modeling.
4. Calculate the potential impact of model implementation on marketing ROI and conversion rate.

## About the Dataset
The dataset was obtained from [Rakamin Academy](https://www.rakamin.com/).

**Description:**

- <code>AcceptedCmp1</code> - 1 if customer accepted the offer in the 1st campaign, 0 otherwise
- <code>AcceptedCmp2</code> - 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
- <code>AcceptedCmp3</code> - 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
- <code>AcceptedCmp4</code> - 1 if customer accepted the offer in the 4th campaign, 0 otherwise
- <code>AcceptedCmp5</code> - 1 if customer accepted the offer in the 5th campaign, 0 otherwise
- <code>Response</code> - 1 if customer accepted the offer in the last campaign, 0 otherwise
- <code>Complain</code> - 1 if customer complained in the last 2 years
- <code>Dt_Customer</code> - date of customer’s enrolment with the company
- <code>Education</code> - customer’s level of education
- <code>Marital</code> - customer’s marital status
- <code>Kidhome</code> - number of small children in customer’s household
- <code>Teenhome</code> - number of teenagers in customer’s household
- <code>Income</code> - customer’s yearly household income
- <code>MntFishProducts</code> - amount spent on fish products in the last 2 years
- <code>MntMeatProducts</code> - amount spent on meat products in the last 2 years
- <code>MntFruits</code> - amount spent on fruits products in the last 2 years
- <code>MntSweetProducts</code> - amount spent on sweet products in the last 2 years
- <code>MntWines</code> - amount spent on wine products in the last 2 years
- <code>MntGoldProds</code> - amount spent on gold products in the last 2 years
- <code>NumDealsPurchases</code> - number of purchases made with discount
- <code>NumCatalogPurchases</code> - number of purchases made using catalogue
- <code>NumStorePurchases</code> - number of purchases made directly in stores
- <code>NumWebPurchases</code> - number of purchases made through the company’s website
- <code>NumWebVisitsMonth</code> - number of visits to the company’s website in the last month
- <code>Recency</code> - number of days since the last purchase
- <code>Z_CostContact</code> - cost to contact customer
- <code>Z_Revenue</code> - revenue after client accepts campaign

**Overview:**

- Dataset contains 2240 rows, 28 features, 1 <code>ID</code> column and 1 redundant <code>Unnamed: 0</code> index column which is removed.
- Dataset consists of 3 data types; float64, int64 and object
- <code>Dt_Customer</code> column could be changed into datetime data type
- Dataset contains 24 Null Values from the <code>Income</code> feature

## Feature Engineering
The following features are extracted from existing default features in order to aid in analysis and modeling.

1. <code>**Total_Spending**</code>:<br>The total of each customer’s spending: sum of MntCoke, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts and MntGoldProducts.
2. <code>**Total_Acc**</code>:<br>The total number of accepted campaigns by each customer, including the response for the last campaign.
3. <code>**Total_Purchases**</code>:<br>The total number of purchases made by each customer.
4. <code>**Total_Children**</code>:<br>The total number of children each customer has.
5. <code>**Conversion_Rate**</code>:<br>Total_Acc divided by number of web visits (NumWebVisitsMonth).
6. <code>**Age**</code>:<br>Age of each customer: 2014-Year_Birth.
7. <code>**Age_Group**</code>:<br>Segmentation of customer ages into 6 groups.
8. <code>**Has_Partner**</code>:<br>Segmentation of Marital_Status: “Yes” if married or engaged, “No” otherwise.

## Data Analysis
### Multivariate (Numerical)
<img src="https://github.com/farrellwahyudi/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Images/numerical_corr.png" alt="Numerical Correlation" style="width:500px;height:500px;"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>Correlation Heatmap of Numerical Features</b>
<br><br>

**Correlation Analysis:**
- **Strong Positive Correlations**: Features like <code>Total_Spending</code> and <code>Total_Purchases</code> have strong positive correlations with various spending categories (<code>MntCoke</code>, <code>MntMeatProducts</code>, etc.), indicating that customers who spend more on these categories tend to spend more overall.
- **Negative Correlations**: <code>NumWebVisitsMonth</code> has negative correlations with several features, suggesting that customers who visit the website more frequently tend to spend less on certain categories.
- **Income Correlations**: <code>Income</code> is positively correlated with most spending categories and <code>Total_Spending</code>, indicating that customers with higher incomes tend to spend more.
- **Recency Correlations**: <code>Recency</code> has weak correlations with most features, suggesting that it doesn't strongly correlate with other features in the dataset.

**Conversion Rate Correlations:**

- **Income** (0.33): There's a moderate positive correlation between <code>Income</code> and Conversion Rate, suggesting that customers with higher incomes tend to have a higher conversion rate.
- **Total_Spending** (0.47): Conversion Rate is positively correlated with <code>Total_Spending</code>, indicating that customers who spend more tend to have a higher conversion rate.
- **NumCatalogPurchases** (0.36): Conversion Rate has a moderate positive correlation with <code>NumCatalogPurchases</code>, implying that customers who make catalog purchases are more likely to have a higher conversion rate.
- **Age** (-0.02): Interestingly, <code>Age</code> has little to no correlation with Conversion Rate, suggesting that even if there is a relationship it is either non-existent or non-linear in nature.
- **Recency** (-0.05): Like <code>Age</code>, <code>Recency</code> also has a very weak correlation with Conversion Rate.
- **Total_Purchases** (0.21): Conversion Rate has a moderate positive correlation with <code>Total_Purchases</code>, indicating that customers with a higher total number of purchases tend to have a higher conversion rate.

### Scatterplot
**Purchases, Spending and Income Correlations:**

<img src="https://github.com/farrellwahyudi/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Images/total_purchases_vs_income.png" alt="Total Purchases vs. Income" style="width:210px;height:210px;"><img src="https://github.com/farrellwahyudi/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Images/total_spending_vs_total_purchases.png" alt="Total Spending vs. Total Purchases" style="width:210px;height:210px;"><img src="https://github.com/farrellwahyudi/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Images/total_spending_vs_income.png" alt="Total Spending vs. Income" style="width:210px;height:210px;">

**Observation:**

The analysis reveals several noteworthy relationships among the variables. Firstly, Total Purchases exhibit a positive correlation with Income. Moreover, Total Spending is also positively correlated with both Total Purchases and Income, suggesting a connection between these economic factors in the dataset.

**Conversion Rate Correlations:**

<img src="https://github.com/farrellwahyudi/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Images/total_purchases_vs_conversion_rate.png" alt="Total Purchases vs. Conversion Rate" style="width:315px;height:315px;"><img src="https://github.com/farrellwahyudi/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Images/total_spending_vs_conversion_rate.png" alt="Total Spending vs. Conversion Rate" style="width:315px;height:315px;"><img src="https://github.com/farrellwahyudi/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Images/income_vs_conversion_rate.png" alt="Income vs. Conversion Rate" style="width:315px;height:315px;"><img src="https://github.com/farrellwahyudi/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Images/age_vs_conversion_rate.png" alt="Age vs. Conversion Rate" style="width:315px;height:315px;">

**Observation:**

The data analysis reveals interesting insights into the correlations between different variables. Total Purchases and Total Spending both display a positive correlation with Conversion Rate, indicating a potential relationship between customer spending and the likelihood of conversion. Additionally, Income exhibits a decent positive correlation with Conversion Rate, while Age shows little to no correlation with Conversion Rate, as evident from the scatterplot.

### Insights
- Total Spending is decently positively correlated with Conversion Rate. This indicates that customers who spend more in total are more likely to convert.
- Income is also positively correlated with Conversion Rate. Higher-income customers may be more likely to convert. 
- The Age & Recency of the customers have little to no relationship with the Conversion Rate.
- Web Purchases, Catalog Purchases, and Store Purchases show high positive correlations with the Conversion Rate. Customers who make purchases through these channels are some what more likely to convert than other channels.
- Various product categories, such as Coke, Meat Products, and Sweet Products, show decent positive correlations with the Conversion Rate. These products are more popular with the customers than others. 
- The number of children is negatively correlated with the Conversion Rate. Customers with more children are less likely to convert. 
- The number of children on the other hand, is decently positively correlated with Deals (discounts) Purchases. Customers with more children are more likely to purchase discounted products.

## Data Preprocessing
### Handling Null Values:
There were 24 null values. All of which were in the <code>Income</code> feature. Therefore null values were imputed using the median of the feature.

### Handling Anomalous Values:
The <code>Age</code> feature contained values above 100. The jump from 74 years old to 114 years old does not make sense. These data points were therefore dropped.

### Handling Outliers:
Since K-Means Clustering will be used to model the clusters, outliers should be removed so that the algorithm won't be drowned out by the outliers. Outliers from the following features were manually trimmed by looking at the boxplot of each feature: <code>MntMeatProducts</code>, <code>Income</code>, <code>NumWebPurchases</code>, <code>MntSweetProducts</code>, <code>NumCatalogPurchases</code>, <code>Total_Spending</code>.

### Dropping Unnecessary Features:
The <code>Unnamed: 0</code> feature is redundant as there was already an <code>ID</code> feature and thus it was dropped. The following features were also dropped as they were unnecessary for the modelling and analysis: <code>Marital_Status</code>, <code>Dt_Customer</code>, <code>Year_Birth</code>, <code>Kidhome</code>, <code>Teenhome</code>, <code>AcceptedCmp1</code>, <code>AcceptedCmp2</code>, <code>AccptedCmp3</code>, <code>AcceptedCmp4</code>, <code>AcceptedCmp5</code>, <code>Response</code>.

### Feature Encoding:
The categorical features remaining was encoded using label encoding. Those features are: <code>Education</code>, <code>Has_Partner</code>, <code>Complain</code>, <code>Age_Group</code>.

### Feature Scaling:
The numerical features were scaled using Scikit-Learn’s StandardScaler() for the purposes of the clustering.

## Modeling
<img src="https://github.com/farrellwahyudi/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Images/modeling_outline.png" alt="Modeling Outline" style="width:620px;height:270px;">

### Feature Selection
Customers will be segmentized by:
- Purchasing Power: <code>Income</code>
- Monetary Value: <code>Tptal_Spending</code>
- Frequency: <code>Total_Purchases</code>
- Activity: <code>NumWebVisitsMonth</code>
- Loyalty: <code>Total_Acc</code>

### Elbow Method
<img src="https://github.com/farrellwahyudi/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Images/elbow_inertia.png" alt="Elbow Method" style="width:620px;height:340px;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>Inertia by Number of Clusters</b>
<br><br>

<p align="center"><img src="https://github.com/farrellwahyudi/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Images/inertia.png" alt="Inertia" style="width:100px;height:190px;"></p> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>Inertia Drop by Number of Clusters (%)</b>
<br><br>

### Silhouette
<img src="https://github.com/farrellwahyudi/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Images/silhouette.png" alt="Silhouette Score" style="width:620px;height:340px;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>Silhouette Score by Number of Clusters</b>
<br><br>

**From the analysis of both the elbow method and the silhouette score, it was decided to divide the customers into 3 clusters.**

### Using PCA to Visualize Clusters
PCA is used to reduce the dimensionality of the data, so that the clusters can be visualized on a lower dimension (2-D). The 3 clusters can be clearly seen below.

<img src="https://github.com/farrellwahyudi/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Images/clusters.png" alt="Clusters" style="width:550px;height:450px;">

## Analysis of Clusters
### Share of Customers
<img src="https://github.com/farrellwahyudi/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Images/customer_share_cluster.png" alt="Share of Customers in Each Cluster" style="width:550px;height:400px;">

The quantity of customers is not evenly distributed among the clusters.
<br><br>
### Each Feature Within Each Cluster
<img src="https://github.com/farrellwahyudi/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Images/boxplot_cluster_analysis.png" alt="Boxplot Cluster Analysis" style="width:750px;height:350px;">

- **Cluster 0**: High value customers, relatively high income, frequent purchases, but less active in terms of visits.
- **Cluster 1**: High potential customers, high income, high spending, loyal in terms of campaign acceptance, but less frequent purchases and less active as well.
- **Cluster 2**: Low value customers, lower income, infrequent purchases, but active in terms of visits.

### Age Group 
<img src="https://github.com/farrellwahyudi/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Images/age_group_cluster.png" alt="Age Group Cluster Analysis" style="width:750px;height:350px;">

- High value cluster consists of mostly older customers with adult and middle aged customers dominating.
- The number of adults and young adults are almost equal in the High potential cluster with both of them dominating.
- Low value cluster consists of mostly adults with young adults in 2nd place.

### Purchases vs. Income
<img src="https://github.com/farrellwahyudi/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Images/total_purchases_vs_income_cluster.png" alt="Purchases vs. Income Cluster Analysis" style="width:550px;height:450px;">

As can be seen above low value customers make less and buy less, while high value and high potential customers make more and buy more, with high potential customers making slightly more money on average.

### Spending vs. Purchases
<img src="https://github.com/farrellwahyudi/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Images/total_spending_vs_total_purchases_cluster.png" alt="Spending vs. Purchases Cluster Analysis" style="width:550px;height:500px;">

High potential customers purchase about the same number of times as high value customers, but the former spend considerably more on their purchases on average.

### Final Analysis
**High Value Cluster**:
- Has 851 customers and consists of 38% of the entire customer base.
- Oldest cluster by age, dominated by middle aged and adult customers.
- Highest spender on gold products on average, beating out even the highest spending cluster. A close 2nd on fish products.
- Most purchases overall on average.
- Least recent cluster (50.3 days)
- "Meat of the sandwich" cluster

**High Potential Cluster**:
- Has 212 customers and consists of 10% of the entire customer base.
- Dominated by young adult and middle aged customers.
- Highest spending cluster overall on average.
- Highest earning cluster overall (Rp. 79,7 million) on average.
- Most web purchases of all the clusters on average.
- Most loyal cluster, accepted more campaigns on average.
- Most recent cluster with an average of 46 days.
- Least active cluster with an average of 3 web visits in the last month.
- Highest conversion rate of all clusters (1 on average).
- Gold mine cluster

**Low Value Cluster**:
- Majority cluster, has 1153 customers and consists of 52% of the entire customer base.
- Youngest cluster by age, dominated by young adults and adults.
- Most children of all the clusters (1.2 children on average).
- Most active cluster with an average of 7 web visits last month.
- Close 2nd for most deals purchases with 2.25 on average.
- Lowest earning cluster overall (Rp. 35,5 million) on average.
- Lowest spending cluster overall on average.
- Least purchases overall on average.
- Least loyal cluster, accepted least number of campaigns on average.
- Lowest conversion rate of all clusters (0.03 on average).
- "Low value high quantity" cluster.

## Recommendations
**High Value Customers:**
- Personalized Marketing: Leverage customer data to create personalized marketing campaigns and product recommendations, especially focusing on products that are favourites (e.g., Gold products and Fish Products).
- Upselling: Identify high-margin products and promote them to this cluster. Upsell premium and gold products to take advantage of their higher spending tendencies.
- Product Bundles: Encourage the purchase of complementary products by offering bundled deals. For instance, if a customer buys meat products, suggest adding fish or sweet products to their cart with a discount.
- Product Expansion: Explore expanding product lines to cater to older demographics and their preferences, as they have a higher mean age.

**High Potential Customers:**
- Loyalty Programs: Since this cluster shows high spending and conversion rates, consider implementing loyalty programs to reward and retain these valuable customers (e.g., exclusive memberships and VIP programs).
- Market Diversification: Explore opportunities to expand into related markets, as these customers have high spending capacities and show a willingness to spend on various categories.
- Exclusive Offers: Offer exclusive, high-end, and limited-edition products to tap into their spending capacity and increase their number of purchases.
- Customer Engagement: Engage with these customers through various channels and maintain a strong online presence, as they tend to make web purchases.

**Low Value Customers:**
- Youth-Centric Products: Given the relatively young age of customers in this cluster, develop products and services that resonate with younger demographics.
- Personalized Web Experience: Utilize data on their frequent web visits to personalize their online shopping experience. Recommend products based on their browsing history and past purchases to increase conversion rates.
- Family-Oriented Marketing: Given the relatively high number of children, Consider bundling products or offering family-oriented deals, as they might be family-oriented shoppers.
- Price-Sensitive Offers: Focus on offering value-for-money deals and discounts, as these customers have lower incomes and tend to buy deals.
- Educational Campaigns: To increase engagement and conversion rates, provide educational content about the benefits of different products. Highlight the nutritional value and diverse uses of sweet products in their daily life.
- Customer Retention: Focus on retaining this customer base by providing excellent customer service and building long-term loyalty.

## Potential Impact
If we focus on the high potential customers and target the campaigns to them exclusively we will see a massive improvement on Conversion Rate (~ 0.75 on average) and by extension marketing Return of Investment (ROI). Where:

$$ROI = {Total Revenue - Total Marketing Cost \over Total Marketing Cost}$$

<br>

**<p align="center">ROI before retargeting ----------> -73.08%</p>**

**<p align="center">ROI after retargeting -----------> 57.1%</p>**

By retargeting the campaigns to "High Potential Customers" we have improved the marketing ROI massively.

<br><br>

<img src="https://verifybee.com/wp-content/uploads/2019/11/Header_7cc3c856f5b86ad98f1232bd17cecaf4.gif" alt="marketing campaign gif" style="width:600px;height:400px;" loop=infinite>
