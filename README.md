# 📧 Spam Detector - Interactive Streamlit App

A professional, interactive web application for spam message detection using Machine Learning.

## ✨ Features

- 🔍 **Single Message Classification**: Instantly classify any message as spam or legitimate
- 📊 **Confidence Scores**: View detailed probability scores for predictions
- 📈 **Model Statistics Dashboard**: Comprehensive analytics and visualizations
- 📜 **Prediction History**: Track recent classifications
- 🎨 **Professional UI**: Clean, modern interface with blue/gray theme
- 📱 **Responsive Design**: Works seamlessly on all devices

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure you have these files in the same directory:**
   - `spam_detector_app.py` (the Streamlit app)
   - `spam_detector_complete.joblib` (your trained model)
   - `spam.csv` (your dataset - for dashboard statistics)

### Running the App

```bash
streamlit run spam_detector_app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`

## 📖 How to Use

### Classify Messages

1. Navigate to "🔍 Classify Message" in the sidebar
2. Enter or paste your message in the text area
3. Click "🔍 Classify" button
4. View the result with confidence scores and probability gauge
5. Check the sidebar for recent prediction history

### View Dashboard

1. Navigate to "📈 Dashboard" in the sidebar
2. Explore:
   - Dataset statistics (total messages, spam/ham counts)
   - Distribution charts (pie chart and message length histogram)
   - Model performance metrics
   - Sample messages from the dataset

## 🎯 Example Messages to Try

**Legitimate (Ham):**
- "Hey, are we still meeting for coffee tomorrow at 3pm?"
- "Can you send me the report by end of day?"

**Spam:**
- "WINNER! You've won $1000000! Click here NOW to claim your prize!"
- "Congratulations! You've been selected for a FREE iPhone. Reply NOW!"

## 📊 Model Information

- **Algorithm**: Multinomial Naive Bayes
- **Accuracy**: Displayed in the sidebar and dashboard
- **Features**: Count of unique words in vocabulary

## 🛠️ Tech Stack

- **Streamlit**: Web app framework
- **scikit-learn**: Machine learning model
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation
- **Joblib**: Model serialization

## 📁 File Structure

```
.
├── spam_detector_app.py          # Main Streamlit application
├── spam_detector_complete.joblib # Trained ML model
├── spam.csv                       # Training dataset
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🎨 UI Features

- **Professional Color Scheme**: Blue (#2c5aa0) and gray tones
- **Interactive Elements**: Buttons, gauges, and charts
- **Responsive Layout**: Adapts to different screen sizes
- **Visual Feedback**: Color-coded results (green for ham, red for spam)
- **Real-time Updates**: Instant classification results

## 🔧 Customization

You can customize the app by modifying:
- **Colors**: Edit the CSS in the `st.markdown()` section
- **Layout**: Adjust column ratios and spacing
- **Features**: Add new visualizations or statistics
- **Theme**: Change from professional to other color schemes

## ⚠️ Troubleshooting

**Model not loading:**
- Ensure `spam_detector_complete.joblib` is in the same directory
- Check file permissions

**Dataset not found:**
- The app will still work for classification
- Dashboard statistics won't be available without `spam.csv`

**Import errors:**
- Run `pip install -r requirements.txt` again
- Check Python version compatibility

## 📝 Notes

- The app caches the model and dataset for better performance
- Prediction history is stored in session state (cleared on app restart)
- All visualizations are interactive (zoom, pan, hover for details)

## 🤝 Contributing

Feel free to enhance the app with:
- Bulk message classification
- Model retraining capabilities
- Export prediction history
- Additional visualizations
- Multi-language support

## 📄 License

This project is open source and available for educational purposes.

---

Built with ❤️ using Streamlit and Machine Learning
