import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. بارگذاری داده‌ها ---
# مطمئن شوید فایل creditcard.csv کنار این اسکریپت باشد
try:
    data = pd.read_csv('creditcard.csv')
except FileNotFoundError:
    print("خطا: فایل creditcard.csv پیدا نشد.")
    print("لطفاً دیتاست را از Kaggle دانلود کنید و کنار این فایل قرار دهید.")
    exit()

print("--- اطلاعات کلی دیتاست ---")
print(data.head())
print("\n--- بررسی مقادیر خالی (Null) ---")
print(data.isnull().sum().max()) # باید 0 برگرداند

# --- 2. بررسی عدم توازن (Imbalance) ---
# این مهم‌ترین بخش است!
print("\n--- توزیع کلاس‌ها (سالم در برابر متقلبانه) ---")
class_counts = data['Class'].value_counts()
print(class_counts)

# محاسبه درصد
total_transactions = len(data)
fraud_percentage = (class_counts[1] / total_transactions) * 100
normal_percentage = (class_counts[0] / total_transactions) * 100

print(f"\nتراکنش‌های سالم (0): {normal_percentage:.2f}%")
print(f"تراکنش‌های متقلبانه (1): {fraud_percentage:.2f}%")

# --- 3. مصورسازی عدم توازن ---
print("\nدر حال ایجاد نمودار توزیع کلاس‌ها...")

plt.figure(figsize=(8, 6))
sns.countplot(x='Class', data=data, palette=['#007bff', '#dc3545'])
plt.title('توزیع کلاس: متقلبانه (1) در برابر سالم (0)')
plt.xlabel('کلاس')
plt.ylabel('تعداد تراکنش')
plt.xticks([0, 1], ['سالم (0)', 'متقلبانه (1)'])
plt.show()

print("نمودار نمایش داده شد. به عدم توازن شدید توجه کنید.")