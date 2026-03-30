---
layout: post
title: "Origami Style Prediction"
date: 2025-03-22
category: data-science 
---

**Project period:** July 2024 - August 2024
            
**Role:**
- Collected and preprocessed the Origami images
- Experimented with various classification models
- Deploy the trained model

**Tools:** Python, opencv, keras, tensorflow, streamlit</p>

# Overview

This is the project where I try to use my Data Science knowledge to solve a Origami question: **"Which origamist your folding style is the most similar?"**

I approached this as an image classification problem, where each image represents an origami model and the label corresponds to the origamist who created it.

# Origami data

For the data, I collected the images from origmists' website, or their social account. We have 16 origamists, which various of style and folding techniques, create a rich variety for the classification tasks. They are: Beth Johnson, Chen Xiao, Choi Ju Yong, Eric Joisel, Gen Hagiwara, Giang Dinh, Hideo Komatsu, Hojyo Takashi, Kaede Nakamura, Kamiya Satoshi, Katsuta Kyohei, Kei Watanabe, Kota Imai, Robert Lang, Shuki Kato, Tran Trung Hieu 

{% include image.html 
   src="/assets/images/origami_style_prediction/origami_models.png" 
   caption="Origamists and their models"
   width="500px"
   class="centered" %}


                    <p>At first, I realize the problems is much more challenge than I expected. Unlike some typical
                        classifcation tasks such as distinguish between dogs and cats by some details like the eyes, the
                        ears, or even the color.
                        , with Origami, everything is more difficult, the topic of authors is not specific to just an
                        animal piece or anything, they can fold everything. That's why the barier between them become
                        more abstract like the style of folding.
                    </p>
                    <p>Beside that, there is a lot of factor of an images which can noised our prediction, like the
                        background, or the watermark, so we have to remove them. Here, I used a U-net model for
                        processing step </p>

                    <p>The result is look like that</p>

                    <div class="figure-container">
                        <figure class="figure-center">
                            <img src="./Images/Archaeopteryx.jpg" alt="plot1" class="clickable-img"
                                style="width:400px;">
                            <figcaption>Original image</figcaption>
                        </figure>
                        <figure class="figure-center">
                            <img src="./Images/archaeo_removed_background.png" alt="plot2" class="clickable-img"
                                style="width:400px;">
                            <figcaption>Processed image</figcaption>
                        </figure>
                    </div>


                    <p>With the model, I have tried with multiple Resnet-based models, and many training methods like
                        transfer learning or training the whole model from scratch, then pick the one which give me the
                        best result in the validation set.</p>
                    <p>Finally, I deployed the model with Streamlit, the simple demo of the model will look like this:
                    </p>
                    <div class="figure-container">
                        <figure class="figure-center">
                            <img src="./Images/normal_UI.png" alt="plot1" class="clickable-img" style="width:400px;">
                            <figcaption>The simple interface</figcaption>
                        </figure>
                        <figure class="figure-center">
                            <img src="./Images/prediction.png" alt="plot2" class="clickable-img" style="width:400px;">
                            <figcaption>Prediction for an image</figcaption>
                        </figure>
                        <figure class="figure-center">
                            <img src="./Images/combined_prediction.png" alt="plot2" class="clickable-img"
                                style="width:300px;">
                            <figcaption>Combined prediction for multiple images</figcaption>
                        </figure>
                    </div>
                    <p>The workflow of the application is simple. You just simply upload an image of a origami model,
                        then the model will make the prediction and extract top 3 similar origamists.
                        When you upload more images, the finally prediction will be combined.
                    </p>

                    <p>With this project, I have learned a lot about methods in image processing, and how we can use
                        many training strategy for a deep learning model.</p>

                </div>
            </section>


        </main>
        <footer class="site-footer">
            <p>&copy; 2025 Cao Khoi Huynh</p>
        </footer>
    </div>
    <script src="../../../js/sidebar.js"></script>
    <script src="../../../js/post-navigation.js"></script>
    <script src="../../../js/modal.js"></script>

</body>

</html>