# ThriftAssist - OCR Phrase Detection

A powerful OCR tool for detecting and annotating phrases in images using Google Cloud Vision API with fuzzy matching support.

## Features

- 🔍 **Multi-orientation text detection** - Handles horizontal, vertical, upside-down, and diagonal text
- 🎯 **Fuzzy phrase matching** - Finds phrases even with OCR errors or variations
- 📦 **Spanning detection** - Matches phrases that span multiple lines
- 🎨 **Visual annotation** - Draws color-coded bounding boxes with smart label placement
- ⚡ **Configurable** - Easy configuration for thresholds, angles, and text filtering

## Project Structure

```
.
├── README.md            # This file
├── requirements.txt     # Python package dependencies
├── thrift_assist/       # Source code for ThriftAssist
│   ├── __init__.py
│   ├── cli.py           # Command-line interface
│   ├── config.py        # Configuration handling
│   ├── detector.py      # Core detection logic
│   ├── drawer.py        # Visual annotation logic
│   └── ocr.py           # OCR processing logic
└── tests/               # Unit tests for ThriftAssist
    ├── __init__.py
    ├── test_detector.py
    ├── test_drawer.py
    └── test_ocr.py
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/thrift_assist.git
   cd thrift_assist
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Google Cloud Vision API credentials:

   - Follow the [Google Cloud Vision API Quickstart](https://cloud.google.com/vision/docs/quickstart-client-libraries) to create a project and obtain credentials.
   - Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of your service account key file:

     ```bash
     export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-file.json"
     ```

## Usage

Run the command-line interface to start detecting phrases in images:

```bash
python -m thrift_assist.cli --help
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and commit them.
4. Push your branch and create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
