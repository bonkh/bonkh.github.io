---
layout: post
title: "Origami Style Prediction"
date: 2024-08-20
category: data-science 
---

*A project to classify origami styles and identify which origamists' style is most similar to my folding technique.*



**Project period:** July 2024 - August 2024
            
**Roles:**
- Collected and preprocessed the Origami images
- Experimented with various classification models
- Deploy the trained model

**Tools:** Python, opencv, keras, tensorflow, streamlit</p>

# **Overview**

Beside pursuing Data Science, I am also an enthusiastic origamist. After nearly a year 
of learning Data Science, I came up with an idea to combine both passions — using my  Data Science knowledge to answer a personal question:

**"Which origamist your folding style is the most similar?"**

I approached this question as an image classification problem, where each image represents an origami model and the label corresponds to the origamist who created it.

# **Origami data**

Firsly, I collected the images from origmists' website, or their social account. We have 16 origamists, which various of style and folding techniques, create a rich variety for the classification tasks. They are: Beth Johnson, Chen Xiao, Choi Ju Yong, Eric Joisel, Gen Hagiwara, Giang Dinh, Hideo Komatsu, Hojyo Takashi, Kaede Nakamura, Kamiya Satoshi, Katsuta Kyohei, Kei Watanabe, Kota Imai, Robert Lang, Shuki Kato, Tran Trung Hieu 

{% include image.html 
   src="/assets/images/origami_style_prediction/origami_models.png" 
   caption="Origamists and their models"
   width="500px"
   class="centered" %}


At first, I realized the problem is much more challenging than I expected. Unlike typical 
classification tasks — such as distinguishing dogs from cats by details like eyes, ears, 
or color — origami is far more difficult. The subject matter of each artist is not limited 
to a specific animal or theme; they can fold anything. This makes the boundary between 
styles more abstract, defined by subtle folding techniques rather than visible features.

Additionally, many factors in the images can introduce noise into our predictions, such 
as the background or watermarks. To address this, I used a **U-Net model** as a 
preprocessing step to remove these distractions.

The result looks like this:

{% include image_slider.html
   id="data-slider"
   images="/assets/images/stock_market/Archaeopteryx.jpg,
           /assets/images/stock_market/archaeo_removed_background.png"
   captions="Original image,
             Processed image" %}


With the model, I have tried with multiple Resnet-based models, and many training methods like transfer learning or training the whole model from scratch, then pick the one which give me the best result in the validation set.

Finally, I deployed the model with Streamlit, the simple demo of the model will look like this:

{% include image_slider.html
   id="data-slider"
   images="/assets/images/stock_market/normal_UI.png,
           /assets/images/stock_market/prediction.png,
           /assets/images/stock_market/combined_prediction.png,
           "
   captions="The simple interface,
             Prediction for an image,
             Combined prediction for multiple images
             " %}

The workflow of the application is simple. You just simply upload an image of a origami model, then the model will make the prediction and extract top 3 similar origamists. When you upload more images, the finally prediction will be combined.

# **Conclusion**
With this project, I have learned a lot about methods in image processing, and how we can use  many training strategy for a deep learning model.
