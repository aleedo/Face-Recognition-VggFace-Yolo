from utils.evaluate import load_model

def predict_web_image(image_url, model=load_model()):
    # Load the image from the web
    response = requests.get(image_url)

    img = Image.open(BytesIO(response.content)).convert('RGB')

    # Preprocess the image
    processed_image = tf.image.resize(img, image_size)
    
#     processed_image = preprocess_input(img)
    # If no face is detected, return None
    if processed_image is None:
        return None

   # Make a prediction using the VGGFace model
    prediction = model.predict(tf.expand_dims(processed_image, axis=0))
    print(prediction)

    # Convert the prediction from an integer to a string label
    predicted_label = inverse_label_map[np.argmax(prediction)]
    
    return predicted_label, np.max(prediction)


if __name__ == '__main__':
    # Test the predict_web_image function on a sample image
    image_url = 'https://cdn.racingnews365.com/Riders/Raikkonen/_570x570_crop_center-center_none/f1_2021_kr_alf_lg.png?v=1643809079'
    predicted_label, confidence = predict_web_image(image_url=image_url)
    print(predicted_label, confidence)