import pandas as pd
import random

def get_stratified_speakers(metadata_path, test_size=5, val_size=5):
    df = pd.read_csv(metadata_path)
    
    # 1. تحليل كل متحدث (كم شعور عنده، وشو جنسه؟)
    spk_stats = df.groupby('speaker_id').agg(
        num_classes=('label', 'nunique'),
        gender=('gender', 'first')
    ).reset_index()
    
    # 2. استخراج المتحدثين "المثاليين" فقط (اللي سجلوا 4 مشاعر كاملة)
    ideal_spks = spk_stats[spk_stats['num_classes'] == 4]
    
    # فصل الذكور (0) والإناث (1) من المتحدثين المثاليين
    males = ideal_spks[ideal_spks['gender'] == 0]['speaker_id'].tolist()
    females = ideal_spks[ideal_spks['gender'] == 1]['speaker_id'].tolist()
    
    # خلطهم عشوائياً لضمان عدم التحيز
    random.seed(42)
    random.shuffle(males)
    random.shuffle(females)
    
    # 3. بناء مجموعة الـ Test (مثلاً: 2 ذكور و 3 إناث)
    test_spks = males[:2] + females[:3]
    
    # 4. بناء مجموعة الـ Val (مثلاً: 2 ذكور و 3 إناث من المتبقين)
    val_spks = males[2:4] + females[3:6]
    
    # 5. بناء مجموعة الـ Train (كل من تبقى + المتحدثين اللي عندهم نقص بالمشاعر)
    all_spks = df['speaker_id'].unique().tolist()
    train_spks = [s for s in all_spks if s not in test_spks and s not in val_spks]
    
    print(f"[CLUSTERER] Train Speakers: {len(train_spks)} (Mixed qualities)")
    print(f"[CLUSTERER] Val Speakers: {len(val_spks)} (Perfect 4-classes, Balanced Gender)")
    print(f"[CLUSTERER] Test Speakers: {len(test_spks)} (Perfect 4-classes, Balanced Gender)")
    
    return train_spks, val_spks, test_spks