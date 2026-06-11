# Allocation Multiple Model - Model 3 Strong NN Streamlit App

This Streamlit app is built for the Model 3 Strong Neural Network / Two-Pass Review allocation system.

## Expected model files
Place the trained `.npz` model files in the same folder as `app.py`:

- `allocate_classifier_model.npz`
- `allocate_regressor_model.npz`
- `allocate_ranker_model.npz`
- `allocate_auxiliary_model.npz`
- `review_pass1_classifier_model.npz`
- `review_pass1_ranker_model.npz`
- `review_classifier_model.npz`
- `review_regressor_model.npz`
- `review_ranker_model.npz`
- `review_auxiliary_model.npz`
- `site802_allocate_specialist_model.npz`
- `site802_review_specialist_model.npz`
- `ak_allocate_specialist_model.npz`

The app also supports split part files if they are listed in `model_part_manifest.json`.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
