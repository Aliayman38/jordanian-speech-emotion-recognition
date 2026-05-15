models/
├── classical.py      ← SVM + KNN + MLP          (MFCC features)
├── wav2vec_clf.py    ← Wav2Vec2                 (deep embeddings)
├── cnn_spec.py       ← CNN                      (spectrogram)
├── fusion_v1.py      ← MFCC + Wav2Vec2           (old paper)
└── fusion_v2.py      ← MFCC + Wav2Vec2 + CNN     (New idea)
