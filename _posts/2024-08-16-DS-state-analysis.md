---
layout: post
title: "2022 Data Science and Machine Learning State Analysis"
date: 2024-08-16
category: data-science 
---

*A project to explored a Kaggle dataset to get insights about people who is working with data in different industries.*
      

**Period:** Nov 2023 - Jan 2024

**Project team:**

- Pham Le Tu Nhi (Team leader)
- Huynh Cao Khoi 
- Hoang Trung Nam 

**Role:**
- Data Preprocessing
- Exploratory Data Analysis (EDA)</li>

**Tool:**
- Data Preprocessing: Numpy, Pandas
- EDA: Plotly, Matplotlib, Seaborn,...</li>

# **Overview**

This is one of my first project about Data Science I finished in the course **Programming for Data Science**. In this course, we concentrate on building up our programming skills, especially in using data processing and analysis skills with Python and some packages like Numpy or Pandas. The final project required our team to make a detail analysis on a specific dataset.

Our team explored the dataset [2022 Kaggle Machine Learning & Data Science Survey](https://www.kaggle.com/competitions/kaggle-survey-2022/data) to know more about people who is working with data in different industries and the best ways for new data scientists like us to break into the field.

# The dataset

**2022 Kaggle Machine Learning & Data Science Survey** is the most comprehensive dataset available on the state of ML and data science. Kaggle Machine Learning & Data Science Survey is a competition held annually from 2017 until 2022 to conduct an industry-wide survey with the aim of presenting a truly comprehensive view of the current state of Data Science and Machine Learning. The survey took place from 09/16/2022 to 10/16/2022, after Kaggle cleaned the data, it received 23,997 responses.

The questions in this survey contains many topics:
- **Biography**: capture personal and demographic information about the participant, such as age, gender, location, and student status.
- **Education**: explore the paricipant learning journey in Data Science: where they studied, which resources were most useful, education level, and research experience.
- **Programming skills & experience**: ask about participant’s technical background — coding experience, languages, tools, and familiarity with machine learning.
- **Employments**: Describes the participant’s current work situation — role, industry, company size, team structure, ML usage, work activities, and income level
- **Other product used (Cloud computing, Data storage...)**: Identifies which cloud, data storage, and related tools the participant uses and how much they have invested in them.

# **Our analysis**

Through the project, each memember pick a topic and deeply dive into it. In my section, I investigated key trends about the data science state among industries, salary ranges and required skills for various data roles and programming language usage patterns.

We can see some analysis result here:

## **The income range distribution**


{% include image_slider.html
   id="icome-slider"
   images="/assets/images/ds_state_analysis/income_distribution.html,
            /assets/images/ds_state_analysis/income_distribtion_by_country.html,
             /assets/images/ds_state_analysis/income_distribtion_by_role.html,
            "
   captions="Participants' current income distribution,
            Participants' current income distribution by country,
            Participants' current income distribution by country,
            " %}
                 
## **Tool set for Data-related roles**

{% include image_slider.html
   id="icome-slider"
   images="/assets/images/ds_state_analysis/sunburst_data_analyst.html,
            /assets/images/ds_state_analysis/sunburst_data_scientist.html,
             /assets/images/ds_state_analysis/sunburst_ml_engineer.html,
            /assets/images/ds_state_analysis/sunburst_research_scientist.html,
            /assets/images/ds_state_analysis/sunburst_statistician.html"
   captions="Data Analyst,
            Data Scientist,
            Machine Learning Engineer,
            Research Scientist,
            Statistician
            " %}
                    
# **Final thoughts**               
Finally, through this project, I have gained my first experience with data science tasks, more clearly know how data science can be applied to solve our problems and strengthen my foundation in data science.