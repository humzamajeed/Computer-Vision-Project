# 📹 How to Add Video to GitHub README

## Step 1: Generate the Video

Run the detection script to create the processed video:

```bash
python run_detection.py
```

This will create `output_detection.mp4` in your project directory.

## Step 2: Upload Video to GitHub

### Option A: Upload via GitHub Web Interface (Easiest)

1. Go to your GitHub repository
2. Click "Add file" → "Upload files"
3. Drag and drop `output_detection.mp4`
4. Commit the file

### Option B: Upload via Git

```bash
git add output_detection.mp4
git commit -m "Add demo video"
git push
```

## Step 3: Add Video to README

Add this HTML5 video tag to your README.md (GitHub supports HTML in markdown):

```html
<video width="800" controls>
  <source src="output_detection.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
```

**Or use a relative path if the video is in a subfolder:**

```html
<video width="800" controls>
  <source src="videos/output_detection.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
```

## Alternative: GitHub Releases

If the video file is too large for the repository:

1. Go to your repository → Releases → "Create a new release"
2. Upload `output_detection.mp4` as a release asset
3. Link to it in your README:

```markdown
[![Watch the demo video](https://img.shields.io/badge/Video-Demo-red)](https://github.com/yourusername/yourrepo/releases/download/v1.0/output_detection.mp4)
```

## Video Specifications

- **Format**: MP4 (H.264 codec)
- **Resolution**: Same as input video
- **FPS**: Same as input video
- **File size**: Depends on video length (GitHub has a 100MB file limit for regular files)

## Tips

- Keep videos under 10MB for faster loading
- Consider compressing the video if it's too large
- Use GitHub Releases for larger videos (>25MB)
- Test the video link after uploading

