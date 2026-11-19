# 🍛 Masala Metrics — Eat desi. Know better.

Masala Metrics is an Indian food image-recognition and nutrition-exploration app built with Streamlit, TensorFlow and a trained EfficientNetB0 model. Upload a food photo, view the model's top match, and explore estimated nutrition values for a reference serving or a portion size you choose.

## Highlights

- Recognises dishes across **96 food categories** using a trained EfficientNetB0 model.
- Displays the leading prediction and the model's five highest output scores.
- Looks up reference calories, protein, carbohydrates, fat and fibre in a nutrition CSV.
- Lets users search the dish directory, sort entries and scale nutrition estimates to a selected serving size.
- Includes clear notes about prediction scores, serving estimates and model limitations.

## Tech stack

- Python
- Streamlit
- TensorFlow / Keras — EfficientNetB0
- NumPy and Pandas
- Pillow

## Project structure

```text
Indian-Food-Nutrition-Analyzer/
├── app.py
├── indian_food_effnetb0.keras
├── food_class_names.json
├── nutrition.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## Run locally

1. Install Python 3.11, 3.12 or 3.13. TensorFlow 2.20.0 is pinned to support the included Keras model format.
2. Clone or download this repository.
3. Create and activate a virtual environment.
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the app:

   ```bash
   streamlit run app.py
   ```

## Model and dataset note

The repository includes the trained model file and the class-name mapping required for inference. The original training images are not included. The app uses a separate CSV of reference nutrition values; these values are estimates and can vary by recipe, ingredients, preparation and serving size.

The model predicts a food category from an image. It does **not** measure the weight of food in a photograph. The displayed model scores are not guaranteed calibrated probabilities. This project is for learning and general information, not medical or dietary advice.

## Acknowledgement

Built as a learning project exploring computer vision, deep learning, data handling and interactive application development.
