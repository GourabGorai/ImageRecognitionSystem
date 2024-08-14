
# Image Recognition

This Flask application allows users to upload images of handwritten characters, register them with actual values, and predict characters from uploaded images using a logistic regression model. The application processes the images, converts them into a grayscale format, and normalizes the pixel values to build a dataset for training the model. Users can also visualize the predicted characters by comparing them with registered images.

Key Features:
Image Upload & Registration: Upload images of handwritten characters, and register the actual character values in a CSV file for future predictions.
Character Prediction: Predict the character from an uploaded image using logistic regression, trained on previously registered images.
Visualization: View plots of images that closely match the predicted character, helping to verify the model's accuracy.
Dynamic Plot Generation: The application generates plots of matched images, allowing users to compare the prediction visually.
Folder Structure:
uploads/: Stores uploaded images.
plots/: Stores plots of images generated during the prediction phase.
How It Works:
Image Registration:

Users can upload an image and specify the actual handwritten character.
The image is converted to grayscale, normalized, and stored in a CSV file along with the specified character.
Prediction:

Users can upload a new image for prediction.
The image is processed, and a logistic regression model predicts the character based on the registered dataset.
The application generates and displays plots of images that match the predicted character.
Visualization:

Plots of the matched images are displayed on the result page, allowing users to compare the prediction visually.
Requirements:
Flask
Matplotlib
Pandas
NumPy
scikit-learn
Pillow
Running the Application:
Install the required Python packages.
Run the application using python app.py.
Access the application via http://localhost:5000/.
This project is a basic yet powerful tool for recognizing handwritten characters and serves as an excellent introduction to machine learning and web development with Flask.


## Acknowledgements

I would like to express my deepest gratitude to Mr. Anirudra Ghosh, my instructor at the National Institute for Industrial Training, for his invaluable guidance and support throughout the duration of this project. His expertise and encouragement were instrumental in the successful completion of this work. I am grateful for the knowledge and insights he shared, which have greatly contributed to my understanding of the subject. Thank you for your mentorship and for providing me with the opportunity to learn and grow under your guidance.


## LINK

https://gourabgorai123.pythonanywhere.com/