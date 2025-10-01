# Understanding the Detection Threshold in ThriftAssist

## What is the Threshold?

The threshold is a percentage (50-100) that controls how strict the text matching algorithm is when searching for phrases in images.

## How It Works

### Matching Process
1. **Exact Match**: If the phrase appears exactly in the text, it gets 100% score
2. **Fuzzy Match**: Uses rapidfuzz library to calculate similarity percentage
3. **Threshold Filter**: Only matches above the threshold are returned

### Score Calculation
```python
# Example scores for searching "Lee Child":
"Lee Child" → 100% (exact match)
"Lee child" → 95% (case difference)
"L. Child" → 85% (abbreviation)
"Lee C." → 75% (partial)
"Le Child" → 80% (OCR error)
"Child Lee" → 70% (word order)
```

## Threshold Settings Guide

### **50-60: Very Lenient**
- **Use for**: Handwritten text, poor image quality, experimental searches
- **Pros**: Finds many variations and partial matches
- **Cons**: Many false positives, noise in results
- **Example**: "Lee Child" matches "Le Chi", "Child L", partial text

### **70-80: Balanced (Recommended)**
- **Use for**: Most book spines, CD covers, general text detection
- **Pros**: Good balance of accuracy and completeness
- **Cons**: May miss some valid variations
- **Example**: "Lee Child" matches "L. Child", "Lee child", OCR errors

### **85-95: Strict**
- **Use for**: High-quality images, when precision is important
- **Pros**: High confidence results, fewer false positives
- **Cons**: May miss valid matches with minor variations
- **Example**: "Lee Child" matches exact text and very close variations only

### **95-100: Very Strict**
- **Use for**: Perfect text, testing, when only exact matches wanted
- **Pros**: Only exact or near-exact matches
- **Cons**: Misses many valid variations
- **Example**: "Lee Child" only matches "Lee Child" and "Lee child"

## Image Type Recommendations

### Book Spines
- **Clear, modern books**: 80-85
- **Older books with wear**: 70-75
- **Damaged/faded spines**: 60-70

### CD/DVD Labels
- **Printed labels**: 85-90
- **Handwritten labels**: 60-70
- **Worn/scratched**: 65-75

### Documents
- **Printed documents**: 85-95
- **Photocopies**: 75-85
- **Handwritten**: 50-65

### Photos of Text
- **Good lighting**: 80-90
- **Poor lighting**: 65-75
- **Angled/distorted**: 60-70

## Technical Details

### Fuzzy Matching Algorithm
Uses `rapidfuzz.fuzz.token_set_ratio()` which:
- Ignores word order
- Handles extra/missing words
- Accounts for character variations
- Returns percentage similarity

### Special Cases
- **Common words** (the, and, of): Use stricter matching to avoid false positives
- **Single words**: Lower threshold tolerance
- **Multi-word phrases**: Standard threshold application
- **Spanning phrases**: Combines scores across multiple text regions

## Troubleshooting

### Too Many False Matches
- **Increase threshold** to 85-90
- Check if search terms are too generic
- Verify image quality

### Missing Valid Matches
- **Decrease threshold** to 65-75
- Check for OCR errors in detected text
- Verify phrase spelling and spacing

### Inconsistent Results
- Try threshold range 70-80 for stability
- Check image resolution and clarity
- Consider text orientation issues

## Examples

```python
# Conservative search (high precision)
results = detect_and_annotate_phrases(
    image_path="clear_bookshelf.jpg",
    search_phrases=["Stephen King", "Harry Potter"],
    threshold=85
)

# Aggressive search (high recall)
results = detect_and_annotate_phrases(
    image_path="blurry_collection.jpg", 
    search_phrases=["Lee Child", "Tom Clancy"],
    threshold=65
)
```
