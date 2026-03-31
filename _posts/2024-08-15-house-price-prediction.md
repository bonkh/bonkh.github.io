---
layout: post
title: "House Price Prediction"
date: 2024-08-15
category: data-science 
---
*A project to discorver the real estate's state and predict the price of apartment in Ho Chi Minh City.*

**Project period:** Nov 2023 – Jan 2024

**Project team:**
- Pham Le Tu Nhi (Team leader)
- Huynh Cao Khoi

**Role:**
- Data Collecting
- Data Preprocessing
- Exploratory Data Analysis (EDA)
- Model Development and Evaluation
- Model Deployment
                        
**Tool:**
- Data Collecting : Selenium, Cloud Scrapper
- Data Preprocessing: Numpy, Pandas, Underthesea,...
- EDA: Matplotlib, Seaborn, Plotly,...
- Model Development and Evaluation: scikit-learn
- Model Deployment: Joblib, Streamlit
                      
# **Overview**

This is the final project for my the first data science course: **Introduction to Data Science**. In this course, I was introduced about the fundamental concepts of data science
and gained my fist knowledge about this field.

The goal of this project was to review and apply everything I had learned by working with the whole process of a data science project. I and my teammate, together, try to solve the problem: **Analyzing of real estate market and predicting property prices.**

**Why did we choose this topic?**

The idea came from a personal experience of my teammate, Nhi. She had just gone through a  exhausting process of finding a new room to rent a few months ago. We realize this can come from the lack of marker insight. That inspired us to explore the real estate market, uncover meaningful patterns, and build a model that could help people make more informed decisions.

# **Collecting dataset**
Firstly, we collected the data from the website [batdongsan.com](https://batdongsan.com.vn ) using web scrapping techniques. My team decided that we will focus on two type of data: **selling data** and **renting data**, each will have diffetent attributes and each member will colllect them separately, make sure that we know the work of each other, also undetstand the whole problem together. The collected data will look like that:

{% include image.html 
   src="/assets/images/house_price_prediction/selling_data.png" 
   caption="The selling real estate data"
   width="600px"
   class="centered" %}
{% include image.html 
   src="/assets/images/house_price_prediction/renting_data.png" 
   caption="The renting real estate data"
   width="600px"
   class="centered" %}

We have over 40.000 samples for selling data and renting data. They contains multiple attributes like: ***Type***, ***Area***, ***Bedrooms***, ***Toilets***, ***Furniture***, ***Floors***, ***Location*** and the target feature ***Price***.

## **EDA**
After that, we do some preprocessing step to make the dataset better, and performed exploratory data analysis to gain insights about the real estate's state

{% include image.html 
   src="/assets/images/house_price_prediction/plot1.png" 
   caption="Distribution of real estate across districts"
   width="500px"
   class="centered" %}

{% include image.html 
src="/assets/images/house_price_prediction/plot2.png" 
caption="Distribution of real estate type across districts"
width="500px"
class="centered" %}

{% include image.html 
src="/assets/images/house_price_prediction/plot3.png" 
caption="Distribution of house price across districts"
width="500px"
class="centered" %}

{% include image.html 
src="/assets/images/house_price_prediction/plot4.png" 
caption="Distribution of apartment price across districts"
width="500px"
class="centered" %}

{% include image.html 
src="/assets/images/house_price_prediction/plot5.png" 
caption="Distribution of mean price of renting real estate type accross district"
width="500px"
class="centered" %}

# **Model Building**
And after that, we build multiple regression model to predict the price of the apartment, for both renting and selling. At this step, we perform the hyperparameter tuning step with **GridSeacch CV**, with the **5-fold cross valiation** training strategy, and chooose the model with the best scrore.

Finally, we deployed our model with a Streamlit application, you can see it here:

{% include image.html 
src="/assets/images/house_price_prediction/demo.png" 
caption="The deployed model"
width="500px"
class="centered" %}

# **Conclusion**
Finally, through this project, I have gained my first experience with data science tasks, more clearly know how data science can be applied to solve our problems and strengthen my foundation in data science.
