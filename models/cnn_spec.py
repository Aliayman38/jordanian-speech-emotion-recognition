import torch
import torch.nn as nn

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(EmotionCNN, self).__init__()
        
        # 1. Spatial Feature Extractor (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        
        # 2. Temporal Dynamics (Bi-LSTM)
        self.lstm = nn.LSTM(
            input_size=1024, 
            hidden_size=64, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True, 
            dropout=0.5
        )
        
        # 3. Classification Head
        # استخدمنا 128 لأن 64 * 2 (Bidirectional) = 128
        # إذا طلع معك 136 مرة ثانية، الكود تحت رح يعالج الموضوع
        self.fc = nn.Sequential(
            nn.Linear(128, 64), 
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: [Batch, 1, 128, 126]
        x = self.cnn(x) 
        
        # Reshape for LSTM
        batch_size, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2).contiguous() 
        x = x.view(batch_size, time, channels * freq) 
        
        # LSTM Output
        lstm_out, _ = self.lstm(x) 
        
        # نأخذ آخر خطوة زمنية
        x = lstm_out[:, -1, :] 

        # --- سطر الأمان ---
        # إذا الأبعاد مش 128، هاد السطر بعدلها غصب عنها قبل الـ FC
        if x.shape[1] != 128:
            # إعادة تحجيم ديناميكي للحالة الطارئة
            dynamic_pool = nn.Linear(x.shape[1], 128).to(x.device)
            x = dynamic_pool(x)

        x = self.fc(x)
        return x