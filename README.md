# Skin Lesion Classifier to Detect Skin Cancer
  In this project, a deep learning-based skill was used to build a skin lesion classifier. This project aims to classify skin lesion for detecting early skin lesion, which could provide a tool for users to detect, including melanoma and other common skin conditions, through image analysis using CNN-based models.

# 📂 Dataset
  The HAM10000 ("Human Against Machine with 10000 training images") dataset was used for CNN model training. It contains 10,015 dermatoscopic images across seven classes of skin lesions:

  Actinic keratoses (akiec)

  Basal cell carcinoma (bcc)

  Benign keratosis-like lesions (bkl)

  Dermatofibroma (df)

  Melanocytic nevi (nv)

  Vascular lesions (vasc)

  Melanoma (mel)

# 🛠️ Tech Stack
  Python

  PyTorch

  NumPy, Pandas

  Matplotlib, Seaborn

  scikit-learn

  Pretrain model has been runned on Kaggle using free GPUs.

# 📊 Results
 98.28% Accuracy, 84.82% Validation Accuracy.

# How to use

  You can classify a new skin image using the saved model like this:
  
  Download resnet50_skin_classifier.pth and main.py
  
  Then run the main.py using streamlit run main.py
