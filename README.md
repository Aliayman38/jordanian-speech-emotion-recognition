models/
├── classical.py      ← SVM + KNN + MLP          (MFCC features)
├── wav2vec_clf.py    ← Wav2Vec2                 (deep embeddings)
├── cnn_spec.py       ← CNN                      (spectrogram)
├── fusion_v1.py      ← MFCC + Wav2Vec2           (old paper)
└── fusion_v2.py      ← MFCC + Wav2Vec2 + CNN     (New idea)

| Model      | Features            | Test Acc (%) |
|------------|---------------------|--------------|
| KNN        | MFCC                | -            |
| SVM        | MFCC                | -            |
| MLP        | MFCC                | -            |
| Wav2Vec2   | Deep Embeddings     | -            |
| CNN        | Spectrogram         | -            |
| Fusion v1  | MFCC + Wav2Vec2     | -            |
| Fusion v2  | MFCC + Wav2Vec2 + CNN | -          |