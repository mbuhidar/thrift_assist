
# Set environment variables to suppress warnings BEFORE importing Google Cloud
import os
import warnings
from contextlib import redirect_stderr
from io import StringIO

# Suppress ALTS warnings and gRPC verbose logging
os.environ['GRPC_VERBOSITY'] = 'NONE'
os.environ['GRPC_TRACE'] = ''
os.environ['GOOGLE_CLOUD_DISABLE_GRPC_FOR_GAE'] = 'true'
os.environ['GLOG_minloglevel'] = '3'

# Suppress all warnings
warnings.filterwarnings("ignore")


def suppress_stderr_warnings():
    """Context manager to suppress stderr warnings during client creation."""
    class FilteredStringIO(StringIO):
        def write(self, s):
            alts_warning = 'ALTS creds ignored' not in s
            log_warning = 'absl::InitializeLog' not in s
            if alts_warning and log_warning:
                return super().write(s)
            return len(s)
    
    return redirect_stderr(FilteredStringIO())


# Import with warnings suppressed
with suppress_stderr_warnings():
    from google.cloud import vision


def run_quickstart() -> vision.EntityAnnotation:
    """Provides a quick start example for Cloud Vision."""

    try:
        # Instantiate client with warnings suppressed
        with suppress_stderr_warnings():
            client = vision.ImageAnnotatorClient()

        # Local image file path
        image_path = "image/gw_cds.jpg"
        
        # Check if local image exists, otherwise use Google Cloud Storage URI
        if os.path.exists(image_path):
            # Read the image file as bytes for local images
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            image = vision.Image(content=content)
            print(f"Using local image: {image_path}")
        else:
            # Fall back to Google Cloud Storage URI
            file_uri = "gs://cloud-samples-data/vision/label/wakeupcat.jpg"
            image = vision.Image()
            image.source.image_uri = file_uri
            print(f"Using remote image: {file_uri}")

        # Performs label detection on the image file
        # response = client.label_detection(image=image)
        # labels = response.label_annotations
        response = client.text_detection(image=image)
        texts = response.text_annotations

        print("Texts:")
        for text in texts:
            print(f'\n"{text.description}"')
            vertices = [
                f"({vertex.x},{vertex.y})" for vertex in text.bounding_poly.vertices
            ]

            print("bounds: {}".format(",".join(vertices)))

        return texts

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
