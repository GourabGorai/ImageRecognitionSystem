from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from PIL import Image
import os
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['PLOTS_FOLDER'] = 'plots/'
app.secret_key = 'supersecretkey'  # Needed for flash messages

# Ensure the upload and plots folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PLOTS_FOLDER'], exist_ok=True)


def viewimg(normalized_pixel_values, plot_path):
    plt.matshow(imgbreak(normalized_pixel_values))
    plt.savefig(plot_path)
    plt.close()


def imgbreak(i):
    imd = []
    count = 0  # Reset count for each row in imagedata
    for j in range(1080):
        dat = []
        for k in range(1080):
            dat.append(i[count])
            count += 1
        imd.append(dat)
    return imd


def imagedatam(imagedata):
    images = []
    for i in imagedata:
        images.append(imgbreak(i))
    return images


# Helper function to create or update the CSV file
def save_image_data(image, actual_value, csv_path):
    # Convert the image to grayscale and normalize to 0-16
    grayscale_img = image.convert('L')
    pixel_values = np.array(grayscale_img).flatten()
    normalized_pixel_values = (pixel_values / 255 * 16).astype(int)

    # Create a DataFrame with the pixel data and actual value
    data = {'pixel_{}'.format(i): normalized_pixel_values[i] for i in range(len(normalized_pixel_values))}
    data['actual_value'] = actual_value
    df = pd.DataFrame([data])

    # Check if the file exists
    if os.path.exists(csv_path):
        # If it exists, append the new data
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        # If it doesn't exist, create it with the new data
        df.to_csv(csv_path, mode='w', header=True, index=False)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'imagefile' not in request.files:
        return redirect(request.url)

    image_file = request.files['imagefile']

    if image_file.filename == '':
        return redirect(request.url)

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    image_file.save(image_path)

    return render_template('preview.html', image_path=image_path)


@app.route('/process', methods=['POST'])
def process():
    image_path = request.form['image_path']
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'ImageData.csv')

    if not os.path.exists(csv_path):
        flash('The CSV file does not exist. Please register some images first.')
        return redirect(url_for('index'))

    data = pd.read_csv(csv_path)
    data = pd.DataFrame(data)
    data.iloc[:,-1]=data.iloc[:,-1].astype(str)
    imagedata = data.iloc[:, :-1].values.tolist()
    target = data['actual_value'].tolist()

    imgdata = imagedatam(imagedata)

    # Making model for Logistic Regression
    X_train, y_train = imagedata, target

    # Exception handling for logistic regression
    try:
        if len(set(y_train)) <= 1:
            raise ValueError("there is only 1 option, please provide more options")

        model = LogisticRegression()
        model.fit(X_train, y_train)
    except ValueError as e:
        flash(str(e))
        return redirect(url_for('index'))

    img = Image.open(image_path)
    img = img.resize((1080, 1080))

    grayscale_img = img.convert('L')
    pixel_values = np.array(grayscale_img).flatten()

    # Normalize grayscale values to 0-16
    normalized_pixel_values = (pixel_values / 255 * 16).astype(int)
    normalized_pixel_values = normalized_pixel_values.tolist()

    predicted_value = model.predict([normalized_pixel_values])[0]
    act = data[data['actual_value'] == predicted_value]

    plot_paths = []
    for index, row in act.iterrows():
        image_pixels = row[:-1].tolist()
        plot_filename = f"{uuid.uuid4().hex}.png"
        plot_path = os.path.join(app.config['PLOTS_FOLDER'], plot_filename)
        viewimg(image_pixels, plot_path)
        plot_paths.append(plot_filename)

    return render_template('result.html', predicted_value=predicted_value, plots=plot_paths)


@app.route('/register', methods=['POST'])
def register_new_image():
    if 'imagefile' not in request.files or 'actual_value' not in request.form:
        return redirect(request.url)

    image_file = request.files['imagefile']
    actual_value = request.form['actual_value']

    if image_file.filename == '' or actual_value == '':
        return redirect(request.url)

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    image_file.save(image_path)

    img = Image.open(image_path)
    img = img.resize((1080, 1080))

    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'ImageData.csv')
    save_image_data(img, actual_value, csv_path)

    flash('Registration completed successfully!')
    return redirect(url_for('index'))


@app.route('/plots/<filename>')
def plot_file(filename):
    return send_from_directory(app.config['PLOTS_FOLDER'], filename)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
