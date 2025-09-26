
import os
import logging

# Suppress ALTS warnings and gRPC verbose logging
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''
os.environ['GOOGLE_CLOUD_DISABLE_GRPC_FOR_GAE'] = 'true'

# Set logging level to suppress warnings
logging.getLogger('google.auth').setLevel(logging.ERROR)
logging.getLogger('google.cloud').setLevel(logging.ERROR)

# Imports the Google Cloud client library
from google.cloud import vision


def run_quickstart() -> vision.EntityAnnotation:
    """Provides a quick start example for Cloud Vision."""

    try:
        # Instantiates a client
        client = vision.ImageAnnotatorClient()

        # The URI of the image file to annotate
        file_uri = "gs://cloud-samples-data/vision/label/wakeupcat.jpg"

        image = vision.Image()
        image.source.image_uri = file_uri

        # Performs label detection on the image file
        response = client.label_detection(image=image)
        labels = response.label_annotations

        print("Labels:")
        for label in labels:
            print(label.description)

        return labels
        
    except Exception as e:
        error_msg = str(e)
        
        # Check if it's a credentials error
        is_creds_error = ("DefaultCredentialsError" in error_msg or
                          "credentials" in error_msg.lower())
        if is_creds_error:
            print("❌ Google Cloud Authentication Error!")
            print("\nTo use Google Vision API, set up authentication:")
            print("1. Create a Google Cloud Project at:")
            print("   https://console.cloud.google.com/")
            print("2. Enable the Vision API for your project")
            print("3. Create a service account and download the JSON key file")
            print("4. Set the environment variable:")
            print("   export GOOGLE_APPLICATION_CREDENTIALS=")
            print("   /path/to/your/service-account-key.json")
            print("\nAlternatively, you can use the local OCR scripts:")
            print("- ocr_annotate_paddleocr.py")
            print("- ocr_annotate_easyocr.py")
            print("- ocr_annotate_pytesseract2.py")
            print("- ocr_annotate_sceneocr.py")
        else:
            print(f"❌ Error running Google Vision API: {e}")
        
        return []


if __name__ == "__main__":
    run_quickstart()