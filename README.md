# ceng-407-408-2022-2023-Speech-Emotion-Recognition
Speech Emotion Recognition

> **Abstract:** Speech emotion recognition is a task of human-computer interaction. People often prefer verbal communication to communicate. It is possible to extract emotion from this communication. Emotion recognition features are extracted from speech signals, features are selected, and emotions are recognized. This report covers our speech emotion recognition project, speech and explains our purpose of doing mood analysis from text. In project development: Python to manage image, audio and text processing. Machine learning algorithms and artificial neural networks will be used to train the model. Librosa, a Python package for music and sound analysis, will also be used.

In this code block, a neural network model is created using the Keras library. The model is a sequential model with multiple dense layers and ReLU activation functions. The number of labels is determined by the shape of the Y variable, which is created by one-hot encoding the target variable y using a label encoder. The model is compiled using the Adam optimizer with a learning rate of 0.0001 and categorical cross-entropy loss. The model is then trained for 100 epochs with a batch size of 64 on the training data and evaluated on the test data. Finally, the trained model is saved to a file named “Model1.h5”.


## Traning
### Dependencies Requirements
```bash
pip install -r requirements.txt
```

RESULTS
<table>
  <tr>
    <th>Method</th>
    <th>Dataset</th>
    <th>Emotions</th>
    <th>Technique</th>
    <th>Accuracy</th>
  </tr>
  <tr>
    <td>Emotion Recognition with Speech</td>
    <td>RAVDEES</td>
    <td>Angry – Disgust – Fear – Happy – Neutral – Sad</td>
    <td>SVM + ANN</td>
    <td>%99</td>
  </tr>
  <tr>
    <td>Emotion Recognition with Speech</td>
    <td>IEMOCAP</td>
    <td>Angry – Excited – Frustration – Happy – Neutral – Sad</td>
    <td>SVM + ANN</td>
    <td>%42</td>
  </tr>
  <tr>
    <td>Emotion Recognition with Speech</td>
    <td>IEMOCAP + RAVDEES</td>
    <td>Angry – Excited – Frustration – Happy – Neutral – Sad</td>
    <td>SVM + ANN</td>
    <td>%54</td>
  </tr>
  <tr>
    <td>Emotion Recognition with Speech</td>
    <td>IEMOCAP</td>
    <td>Angry – Excited – Frustration – Happy – Neutral – Sad</td>
    <td>ANN</td>
    <td>%33</td>
  </tr>
  <tr>
    <td>Emotion Recognition with Speech</td>
    <td>IEMOCAP</td>
    <td>Angry – Disgust – Fear – Happy – Neutral – Sad</td>
    <td>CNN + RESNET</td>
    <td>%30</td>
  </tr>
  <tr>
    <td>Emotion Recognition with Speech</td>
    <td>IEMOCAP</td>
    <td>Angry – Disgust – Fear – Happy – Neutral – Sad</td>
    <td>CNN + RESNET + ALEXNET</td>
    <td>%24</td>
  </tr>
  <tr>
    <td>Emotion Recognition with Speech</td>
    <td>IEMOCAP</td>
    <td>Angry – Disgust – Fear – Happy – Neutral – Sad</td>
    <td>DNN + CRNN</td>
    <td>%32</td>
  </tr>
  <tr>
    <td>Emotion Recognition with Text</td>
    <td>IEMOCAP</td>
    <td>Angry – Excited – Frustration – Happy – Neutral – Sad – Fear - Disgust</td>
    <td>BERT + ANN</td>
    <td>%26</td>
  </tr>
  <tr>
    <td>Emotion Recognition with Text</td>
    <td>IEMOCAP</td>
    <td>Angry – Excited – Frustration – Happy – Neutral – Sad</td>
    <td>BERT + ANN</td>
    <td>%82</td>
  </tr>
</table>
