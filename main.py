from utils.load_preprocess import get_data, get_train_val_test_splits
from utils.train import fit_model
from utils.evaluate import eval_model
from utils.predict import predict_web_image


def main(*args, **kwargs):
    ds, _ = get_data()
    train_dataset, val_dataset, test_dataset = get_train_val_test_splits(ds)
    _ = fit_model(train_dataset, val_dataset)
    test_loss, test_accuracy = eval_model(test_dataset)
    predicted_label, confidence = predict_web_image(image_url=image_url)


if __name__ == "__main__":
    image_url = "https://cdn.racingnews365.com/Riders/Raikkonen/_570x570_crop_center-center_none/f1_2021_kr_alf_lg.png?v=1643809079"
    print(main(image_url=image_url))
