---
layout: post
title: "Language identification models evaluation"
date: 2025-03-22
category: data-science 
---

**Project period:** Septemper 2025 - March 2026</p>

**Team size:**
                        
- Hoan Nguyen
- Huynh Cao Khoi

**Role**                       
- Data Collecting
- Data Preprocessing
- Exploratory Data Analysis (EDA)
- Model Evaluation
- Evaluation result analysis

**Tools**
- Data Collecting : Hugging Face API
- Data Preprocessing: Polars, sentence_splitter, pyspark,unicodedata, semhash, cleanlab,...
- EDA: Matplotlib, Seaborn,...
- Model Evaluation: pytorch
- Evaluation result analysis: 


# **Overview**
- This project is the work in my intership for Mr.Hoan Nguyen. In this project,I try to research and find the current state of the language indentification problem, which is an important element in many language models.

In this project, I have to do these bellow main tasks:

- Find and study about text datasets for language identification task.
- Preprocess and combined all datasets together.
- EDA the dataset
- Evaluate models in [this collection](https://huggingface.co/collections/hoan/languages-identification) (combined with state-of-the-arts) in many metrics (even the runtime performance) with the combined dataset.
- Analyze the results 

# **Task 01: Collect text dataset**

The full list of text dataset can be found in [this spreedsheet](href="https://docs.google.com/spreadsheets/d/1G12FaSMelNX87dclhm9dE3d5ZRO99B2hoFE1ufB2Zvg/edit?usp=sharing)

These datasets consist of training text dataset, or benchmark dataset, which is published mostly in recent years, cover a wide range of topics and domains

The collected data is in this format:

{% include image.html 
   src="/assets/images/lid_evaluation/raw_data.png" 
   caption="Raw data"
   width="500px"
   class="centered" %}


As you can see, these dataset was consist of two main attributes:
- **Text:** This is the main text
- **Language:** The language code, which normally in the format language code(ISO 639-3) - language script(ISO 15924)

# **Task 02: Merge and preprocess the combined text datasets**
- After downloading the dataset from multiple sources, I performed some simple preprocessing steps in each dataset.

{% include image.html 
   src="/assets/images/lid_evaluation/raw_data_process.png" 
   caption="Raw data processing pipeline"
   width="500px"
   class="centered" %}

In the handling long text step, I used [sentence-split](https://pypi.org/project/iges-sentence-splitter/) package to split the text into sentences more accurately.

After that, I combined all the sources into one dataset, using this process:

{% include image.html 
   src="/assets/images/lid_evaluation/combined_data_process.png" 
   caption="Processing pipeline for combined data"
   width="500px"
   class="centered" %}


With the combined dataset, I performed these main steps to clean it:

**Step 01: Clean the programming language pattern**

Programming language is not the natural language, I considered it as noise in a LID dataset

I simply used [re](https://docs.python.org/3/library/re.htmlz) package to find the programming language pattern inside the text. The text containing mostly programming language pattern will be removed. In others, I cleand all the possible patterns.

{% include image.html 
   src="/assets/images/lid_evaluation/programming_language_removal.png" 
   caption="Progamming language removing pipeline"
   width="500px"
   class="centered" %}


**Step 02: Detect abnormal symbols in texts:**

Symbols such as #, ^, ~, -, and non-Unicode characters negatively affect text quality and do not contribute useful information for language identification.

To detect the abnormal symbols, I used [unicodedata](https://docs.python.org/es/3.13/library/unicodedata.html) package to detect the character which is not ịn types: letter, mark, number

{% include image.html 
   src="/assets/images/lid_evaluation/symbols_removal.png" 
   caption="Abnormal symbols removing pipeline"
   width="500px"
   class="centered" %}

**Step 03: Clean the text with high ratio of number**

Texts consisting mostly numbers, can be math calculation, equation, or numeric tables, they should also be removed.

{% include image.html 
   src="/assets/images/lid_evaluation/digit_process.png" 
   caption="Digit processing pipeline"
   width="500px"
   class="centered" %}
                  

**Step 04: Select mostly character texts**

After multiple cleaning steps, some texts may become empty or just contain very little information left, now we need to keep only texts that contain a sufficient proportion of
informative characters.

{% include image.html 
   src="/assets/images/lid_evaluation/letter_ratio.png" 
   caption=""
   width="500px"
   class="centered" %}

The final dataset is stored in parquet files. The storage is significantly reduced compared with the tsv files in the previous step.

**Subsampling the data using SemHash and Cleanlab**

The data now contains about 48,320,698 samples, which is too large to run the evaluation step on the current device. Therefore, the data must be downsampled, retaining only the highest-quality samples.

Here, I used [SemHash](https://github.com/MinishLab/semhash), a lightweight, multimodal library, used for semantic deduplication, outlier filtering, and representative sample selection.

SemHash was applied separatedly to each language:

{% include image.html 
   src="/assets/images/lid_evaluation/semhash_apply.png" 
   caption="SemHash application pipeline"
   width="500px"
   class="centered" %}


Finally, the data is stored in many parquet files, one per each language. For more details, visit [this notebook](https://www.kaggle.com/code/caokhoihuynh/semhash-apply)
                   
In a language dataset, an important issue is language misslabeling. The incorrect labeled samples can significantly affect the training process, and negativly impact the model performance.

To solve this problem, I used [Cleanlab](https://github.com/cleanlab/cleanlab) a library which works with any ML model by analyzing model outputs like predicted probabilities and feature embeddings to identify problems such as label errors, outliers, near duplicates, and other data quality issues.

In Cleanlab, there are two important elements:

- The embeder: Create the embedding vectors of the given text
- Classifier: Trained from the data, to create the predicted language probabilities for a given text


After trying with multiple setting of embedder and classifier, I have these main settings. The detailed implementation can be found in [this notebook](https://www.kaggle.com/code/caokhoihuynh/check-the-data-issue-cleanlab)
- Embeder: FastText - Classifier: Logistic regression, API:cleanlab.classification.CleanLearning </li>
                        <li>Embeder: Transformer + FastText - Classifier: XGB classifier, API:
                            cleanlab.classification.CleanLearning </li>
                        <li>Embeder: FastText - Classifier: Logistic regression, API: cleanlab.filter.find_label_issues
                        </li>
                        <li>Embeder: Transformer - Classifier: Logistic regression, API:
                            cleanlab.filter.find_label_issues </li>
                    </ul>
Different settings will result in a different label-cleaned versions of dataset. From that, I ensembled all of them, using a simple strategy that a text, which will have a highly confident about the label quality, if it appear in all versions. The detailed implementation can be found in [this notebook](https://www.kaggle.com/code/caokhoihuynh/ensemble-data-02).

After all of these steps, I have a cleaned version of the dataset, containing about 1,5 million samples. And it still is not completely clean. 🥲🥲🥲


# **Task 03: Doing EDA for the collected dataset**

The detailed analysis on the dataset can be found in [this notebook](https://www.kaggle.com/code/caokhoihuynh/eda-multilingual-dataset)

Some key features about the dataset can be listed:

{% include image_slider.html
   id="eda-slider"
   images="/assets/images/lid_evluation/source_similarity.png,
           /assets/images/lid_evluation/length_distribution.png,
           /assets/images/lid_evluation/word_count_distribution.png"
   captions="Source similarity in Jaccard distance,
             Text length distribution,
             Word count distribution" %}

# **Task 04: Evaluate the cleaned dataset with models**

Here is the list of model I evaluated, also consider the supporting language list for all these models:
- [papluca/xlm-roberta-base-language-detection](https://huggingface.co/papluca/xlm-roberta-base-language-detection): 20 possible labels
- [WordLlama Detect](https://github.com/dleemiller/WordLlamaDetect): 146 possible labels
- [julien-c/fasttext-language-id](https://huggingface.co/julien-c/fasttext-language-id): 175 possible labels
- [NeuML/language-id](https://huggingface.co/NeuML/language-id): 175 possible labels
- [laurievb/OpenLID-v2](https://huggingface.co/laurievb/OpenLID-v2): 194 possible labels
- [alexneakameni/language_detection](https://huggingface.co/alexneakameni/language_detection): 195 possible labels
- [facebook/fasttext-language-identification](https://huggingface.co/facebook/fasttext-language-identification): 211 possible labels
- [epfl-nlp/ConLID](https://github.com/epfl-nlp/ConLID): 1869 possible labels
- [cis-lmu/glotlid](https://huggingface.co/cis-lmu/glotlid): 1873 possible labels
 
We can see the **papluca** model only supports a limited set of languages (20 languages), 
while **conlid** and **glotlid** can support more than 1800 languages. Other models support 
around 100–200 languages.

To ensure a fair evaluation between models, we consider three scenarios:

- **Scenario 1:** Evaluate all models, except papluca, on an 83-language dataset (the overlap of supported languages across all other models).
- **Scenario 2:** Evaluate all models, including papluca, on a 17-language dataset.
- **Scenario 3:** Evaluate only conlid and glotlid on a 1810-language dataset.

For each scenario, I evaluated each language separately, recording:
- The accuracy of each model per language
- The total inference time per language
- The inference time per sample per language

Additionally, I stored the detailed prediction results of all models for each sample, for future analysis.

The full evaluation code can be found in the notebooks: 
[CPU evaluation](https://www.kaggle.com/code/caokhoihuynh/run-evaluation-cpu) and 
[GPU evaluation](https://www.kaggle.com/code/caokhoihuynh/run-evaluation-gpu)

## Task 05: Analyze the Evaluation Results

The detailed analysis can be found in [this notebook](https://www.kaggle.com/code/caokhoihuynh/evaluate-results-analysis).

Here, I will summarize some key insights:

### 1. Metrics

I evaluated all models using two metrics: **accuracy** and **F1-macro score**.

{% include image.html 
   src="/assets/images/lid_evaluation/metric_results.png" 
   caption="RAM usage for CPU evaluation"
   width="500px"
   class="centered" %}

In Scenario 1, the evaluation was conducted on around 100 of the most popular languages.
Among the evaluated models, **Conlid** and **lmu_glotlid** achieved the highest performance 
in both accuracy and F1 score. These models also have the widest language coverage, with 
more than 1800 languages evaluated in Scenario 3. However, this high performance was not 
consistently maintained across both scenarios. One possible reason is the evaluation data 
itself — many low-resource languages (~700 languages) have only one sample, which can lead 
to unfair or unstable results.

In Scenario 2, where the evaluation focused on the top 17 most popular languages, the 
**papluca** model — which supports only 20 languages — achieved the highest performance. 
This suggests the model is primarily trained on popular languages and performs well in 
simpler, high-resource settings.

Additionally, the **neuml_langid** model showed very poor performance in Scenario 1. 
Although its performance improved in Scenario 2, it still remained significantly lower 
than the other models.

### 2. Memory Usage

Memory usage is a key factor when selecting a language identification model. Ideally, 
we want a model that is lightweight, yet achieves strong performance and fits practical 
deployment requirements.


{% include image_slider.html
   id="eda-slider"
   images="/assets/images/lid_evluation/RAM_usage.png,
           /assets/images/lid_evluation/VRAM_usage.png"
   captions="RAM usage for CPU evaluation,
             VRAM usage for GPU evaluation" %}

 
Look at the RAM usage, we observe that evaluated models requires from hundreds to thousands of megabytes. Only a few transformer-based model can support the GPU execution. Therefore, most evaluations are performed on the CPU. This als reflects a trend in model development nowadays, where models are become more lightweight and optimized for CPU-based inference.

In more detail, I used the pareto plot to show the relation between memory usage and the model accuracy. The analysis focuses on Scenario 01 and Scenario 02, where multiple models are compared under the same evaluation settings.
                   
{% include image_slider.html
   id="eda-slider"
   images="/assets/images/lid_evluation/scen_01_ram_vs_metrics.png,
           /assets/images/lid_evluation/scen_02_ram_vs_metrics.png"
   captions="Pareto plot for RAM usage vs metrics in scenario 01,
             Pareto plot for RAM usage vs metrics in scenario 02" %}

 
From the results, **Conlid** achieves the highest accuracy while using an acceptable 
amount of memory. Additionally, **alexneakameni** is the most lightweight model, with 
competitive accuracy. **llamadetect** also offers a strong balance between low memory 
consumption and performance.

### 3. Throughput

Throughput is measured as the number of samples a model can process per second. 
This is an important factor for many real-world applications that require a fast and 
accurate language identification module.

{% include image_slider.html
   id="eda-slider"
   images="/assets/images/lid_evluation/scen_01_thoughput_vs_metrics.png,
           /assets/images/lid_evluation/scen_02_thoughput_vs_metrics.png"
   captions="Pareto plot for RAM usage vs metrics in scenario 01,
             Pareto plot for RAM usage vs metrics in scenario 02" %}


 In terms of throughput, **Conlid**, despite achieving the highest accuracy, is the 
slowest model. This suggests that Conlid is best suited for scenarios where accuracy 
is the top priority, rather than memory efficiency or inference speed.

The **julien_fasttext** model, which has very low memory consumption, achieves the 
highest throughput. However, its overall accuracy is not particularly strong. Therefore, 
it is a suitable option for tasks with strict speed and resource constraints.

Additionally, the **llamadetect** model offers a well-balanced trade-off across accuracy, 
memory usage, and throughput, making it a strong candidate for a wide range of practical 
applications.

## Conclusion

Through this project, I gained valuable knowledge about current language identification 
methods and hands-on experience working in the NLP field.