---
title: "EasyTopics"
excerpt: "A web app to detect, analyze and visualize topics from your documents using BERTopic Wikipedia<br/><em>Keywords: NLP, Transformers</em><br/><br/><img src='/images/yourdocuments.png'>"
collection: portfolio
---

A web app to detect, analyze and visualize topics from your documents using BERTopic. 

Developers: Chris Tam, Ian Bulovic, Nhi Le (Brandeis University)

[**GitHub Repo**](https://github.com/halannhile/topic-modeling-app/blob/master/README.md)

[**Link to app**](https://easytopics.streamlit.app/)

[**Project report**](https://github.com/halannhile/topic-modeling-app/blob/master/static/COSI-217b%20Final%20Report.pdf)


## App overview

Note: Some of the pages in the app might take a few minutes to load because of the underlying BERTopic model.

### Homepage

This page shows all the documents you have uploaded, classified into two upload types: documents (for individual file uploads), and dataset (for zip uploads). 

You can filter and delete documents by various criteria.

<img src="/images/your_documents.png" width="800">

### Upload Documents page

You can either: 

1. Upload a single document and run a pretrained topic model on it ([BERTopic Wikipedia](https://huggingface.co/MaartenGr/BERTopic_Wikipedia) in this case):

<img src="/images/upload_documents.png" width="250">


2. Upload a dataset in the form of a zip folder and train a topic model. Note: this may take a while depending on the size of your dataset.

<img src="/images/upload_dataset_2.png" width="300">

### Topic Modeling Results page 

On this page, you are given the option to visualize topic modeling results on document uploads or dataset uploads, using either the pretrained BERTopic Wikipedia model, or your own model.

<img src="/images/document_viz.png" width="800">

### Custom Model Visualizations page 

You can also view different topic visualizations of your trained model: 

<img src="/images/embedding_space.png" width="500">

<img src="/images/similarity_matrix.png" width="300">

<img src="/images/score.png" width="300">