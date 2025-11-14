import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# --- 1. بارگذاری داده‌ها ---
try:
    data = pd.read_csv('creditcard.csv')
except FileNotFoundError:
    print("خطا: فایل creditcard.csv پیدا نشد.")
    exit()

print("داده‌ها با موفقیت بارگذاری شدند...")

# --- 2. آماده‌سازی داده‌ها ---
# ستون 'Time' و 'Amount' معمولاً تأثیر زیادی ندارند (چون V1 تا V28 از قبل مهندسی شده‌اند)
# اما ما آنها را نگه می‌داریم تا مدل پیچیده‌تر نشود.
X = data.drop('Class', axis=1) # تمام ستون‌ها بجز هدف
y = data['Class']              # ستون هدف (0 یا 1)

# --- 3. تقسیم داده‌ها (قبل از SMOTE) ---
# داده‌ها را به دو بخش تست (30%) و آموزش (70%) تقسیم می‌کنیم
# ما باید SMOTE را فقط روی داده‌های آموزشی اعمال کنیم، نه داده‌های تست!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"اندازه داده آموزشی (قبل از SMOTE): {len(y_train)}")
print(f"تعداد تقلب در داده آموزشی (قبل از SMOTE): {sum(y_train)}")

# --- 4. اعمال SMOTE (جادوی تعادل‌سازی) ---
print("\nدر حال اعمال SMOTE برای متعادل‌سازی داده‌های آموزشی...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"\nاندازه داده آموزشی (بعد از SMOTE): {len(y_train_resampled)}")
print(f"تعداد تقلب در داده آموزشی (بعد از SMOTE): {sum(y_train_resampled)}")
print("داده‌های آموزشی اکنون متعادل شدند.")

# --- 5. آموزش مدل ---
# ما از یک مدل ساده Logistic Regression استفاده می‌کنیم
print("\nدر حال آموزش مدل Logistic Regression...")
model = LogisticRegression(solver='liblinear', max_iter=200) # max_iter برای همگرایی بهتر
model.fit(X_train_resampled, y_train_resampled)

# --- 6. ارزیابی مدل ---
print("مدل آموزش دید. در حال ارزیابی روی داده‌های تست (که SMOTE روی آن اعمال نشده)...")
y_pred = model.predict(X_test)

# --- 7. نمایش نتایج ---
print("\n--- نتایج نهایی ارزیابی مدل ---")
print(f"دقت کلی (Accuracy): {accuracy_score(y_test, y_pred):.4f}")

print("\n--- ماتریس درهم‌ریختگی (Confusion Matrix) ---")
print("(نشان می‌دهد مدل چند تقلب را درست و چند را اشتباه تشخیص داد)")
print(confusion_matrix(y_test, y_pred))

print("\n--- گزارش طبقه‌بندی (Classification Report) ---")
print("(مهم‌ترین بخش: به Recall برای کلاس 1 نگاه کنید)")
print(classification_report(y_test, y_pred, target_names=['سالم (0)', 'متقلبانه (1)']))