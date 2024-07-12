# Food Classifier 101

This project aims to classify food images into one of 101 different food categories using deep learning models. We utilized the Food-101 dataset and trained models to achieve accurate predictions.

## Dataset

We used the Food-101 dataset, which consists of images from 101 food categories. Due to time constraints, we trained our models on a subset comprising 20% of the dataset.

## Models Used

We experimented with two models:
- EfficientNet-B2
- EfficientNet-B3

## Model Comparison

We compared the models based on the following metrics:
- Training time
- Model size
- Accuracy

After evaluation, we selected EfficientNet-B3 for deployment due to its superior performance in terms of accuracy.

## Deployment

The chosen model, EfficientNet-B3, is deployed using the Gradio SDK at Hugging Face. You can interact with the deployed model [here](https://huggingface.co/spaces/Zeyad-Sayed/Food-Classifier-101).

## Notebook and Code

For detailed implementation and code, refer to our Kaggle notebook available [here](https://www.kaggle.com/code/zeyadsayedadbullah/food-classifier-101/).

## Usage

To use the model:
1. Visit the deployment link provided.
2. Upload an image of food from one of the 101 categories.
3. Get real-time predictions on the type of food.

## Contributors

- Zeyad Sayed (@zeyadsayedadbullah)

