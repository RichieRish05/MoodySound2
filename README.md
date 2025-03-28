# MoodySound

MoodySound is a deep learning project that predicts the mood of songs using spectrograms. The model outputs an 8-dimensional normalized mood vector representing different emotional aspects of music.

## Mood Dimensions

The model predicts the following mood dimensions:
- **Danceability**: How suitable the track is for dancing
- **Acoustic**: Presence of acoustic instruments and natural sounds
- **Aggressive**: Energy and intensity level of the track
- **Electronic**: Presence of electronic/synthetic elements
- **Happy**: Overall positivity and upbeat nature
- **Party**: Suitability for party/celebration settings
- **Relaxed**: Calmness and tranquility level
- **Sad**: Melancholic and emotional content

## Project Structure

### Dataset Creation Pipeline
1. **Data Collection**
   - Extracted high-level mood features from [AcousticBrainz](https://acousticbrainz.org/) dataset
   - Generated spectrograms from audio files
   - Uploaded final dataset to AWS S3

2. **Model Loader**
   - Downloads trained model from AWS S3 bucket
   - Used for local model testing and inference
   - Handles model weight loading and configuration

3. **Model Production**
   - Performed hyperparameter optimization using Ray Tune
   - Conducted distributed training across multiple GPUs
   - Generated final production model with optimized parameters
   - Exported and versioned trained model to AWS S3

## Acknowledgements

- [AcousticBrainz](https://acousticbrainz.org/) for providing the comprehensive music dataset
- UCI Computer Science Department
- The open-source deep learning community for various tools and libraries
- ZOT ZOT ZOT! 🐜

