# Offline Validation Sample Data

Put your labeled sample images in:

```text
data/offline_sample/images
```

You can use either one combined `labels.json` file or one JSON file per image in:

```text
data/offline_sample/labels
```

For one JSON file per image, name the label file with the same stem as the image:

```text
images/example_001.jpg
labels/example_001.json
```

Example:

```json
{
  "name": "my_image.jpg",
  "labels": [
    {
      "category": "person",
      "box2d": {"x1": 25, "y1": 40, "x2": 80, "y2": 160}
    }
  ]
}
```

Use these paths in Streamlit:

```text
Labels Path: data/offline_sample/labels
Image Directory: data/offline_sample/images
```

You can also use:

```text
Labels Path: data/offline_sample/labels.json
Image Directory: data/offline_sample/images
```
