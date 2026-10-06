"""
โมเดล KNN จำแนกผื่น 3 ชนิด: ผดผื่นร้อน / ผื่นอีสุกอีใส / ผื่นโรคมือเท้าปาก
- Features: 11 ลักษณะที่เห็นในรูป (0 = ไม่มี, 1 = มี)
- แบ่งข้อมูล: train 80% / test 20% (stratify)
- KNN: k = 5, distance = Hamming

วิธีใช้:
1) ติดตั้งไลบรารี:  pip install pandas scikit-learn xlrd openpyxl
2) วางไฟล์ Excel ไว้โฟลเดอร์เดียวกับไฟล์นี้ แล้วแก้ DATA_FILE ให้ตรงชื่อไฟล์
3) รัน:  python knn_model.py
"""
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------- ตั้งค่า ----------------
DATA_FILE = "dataset.xls"      # ชื่อไฟล์ Excel (.xls หรือ .xlsx)
LABEL_COL = "คำตอบ (label)"    # ชื่อคอลัมน์คำตอบ
K = 5                          # จำนวน nearest neighbors
RANDOM_STATE = 42              # ใช้ค่าเดิมเพื่อให้แบ่งข้อมูลได้เหมือนเดิมทุกครั้ง

# ---------------- 1) โหลดข้อมูล ----------------
df = pd.read_excel(DATA_FILE)
if "รูป" not in df.columns:                      # ถ้าไม่มีคอลัมน์ชื่อรูป ให้สร้างเลขรูปให้
    df.insert(0, "รูป", [f"รูปที่ {i+1}" for i in range(len(df))])

X = df.drop(columns=["รูป", LABEL_COL])          # features
y = df[LABEL_COL]                                # label
FEATURES = list(X.columns)

print(f"จำนวนรูปทั้งหมด: {len(df)} รูป, จำนวน features: {len(FEATURES)}")
print(y.value_counts().to_string(), "\n")

# ---------------- 2) แบ่ง train / test ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print(f"Training set: {len(X_train)} รูป | Test set: {len(X_test)} รูป\n")

# ---------------- 3) เปรียบเทียบค่า k ด้วย 5-fold cross-validation (ใช้ train เท่านั้น) ----------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
print("Cross-validation accuracy บน training set:")
for k in [1, 3, 5, 7, 9]:
    score = cross_val_score(KNeighborsClassifier(n_neighbors=k, metric="hamming"),
                            X_train, y_train, cv=cv).mean()
    mark = "  <- ค่าที่ใช้" if k == K else ""
    print(f"  k = {k}: {score:.1%}{mark}")
print()

# ---------------- 4) Train โมเดลด้วย k = 5 ----------------
model = KNeighborsClassifier(n_neighbors=K, metric="hamming")
model.fit(X_train, y_train)

# ---------------- 5) ทดสอบกับ test set ----------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"ความถูกต้องในการทดสอบ (Accuracy): {acc:.0%}  ถูกต้อง {(y_test == y_pred).sum()}/{len(y_test)} รูป")
print(f"(ผลจาก Test Set จำนวน {len(y_test)} รูปเท่านั้น ไม่ได้รับรองว่าจะถูกต้อง {acc:.0%} ในทุกกรณี)\n")

print("Confusion matrix (แถว = ชนิดผื่นจริง, คอลัมน์ = ผลที่ระบบจำแนก):")
cm = pd.DataFrame(confusion_matrix(y_test, y_pred, labels=model.classes_),
                  index=model.classes_, columns=model.classes_)
print(cm.to_string(), "\n")

print("Classification report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("ผลรายรูปใน test set:")
result = pd.DataFrame({"รูป": df.loc[X_test.index, "รูป"],
                       "ชนิดผื่นจริง": y_test, "ผลที่ระบบจำแนก": y_pred})
result["ผล"] = ["ถูกต้อง" if a == b else "ไม่ถูกต้อง" for a, b in zip(result["ชนิดผื่นจริง"], result["ผลที่ระบบจำแนก"])]
print(result.to_string(index=False), "\n")


# ---------------- 6) ฟังก์ชันจำแนกรูปใหม่ ----------------
def predict_rash(features_seen):
    """
    features_seen: list ของชื่อ feature ที่เห็นในรูป เช่น ["ตุ่มน้ำบนฐานสีแดง", "ผื่นหลายระยะในรูปเดียว"]
    คืนค่าผลการจำแนก และแสดง 5 รูปตัวอย่างที่ใกล้เคียงที่สุดที่ใช้ vote
    """
    unknown = [f for f in features_seen if f not in FEATURES]
    if unknown:
        raise ValueError(f"ไม่รู้จัก feature: {unknown}\nfeature ที่ใช้ได้: {FEATURES}")

    x = pd.DataFrame([[1 if f in features_seen else 0 for f in FEATURES]], columns=FEATURES)
    pred = model.predict(x)[0]

    dist, idx = model.kneighbors(x)                 # distance แบบ Hamming เป็นสัดส่วน (0–1)
    print(f"ลักษณะผื่นที่พบ: {features_seen}")
    print(f"ระบบจำแนกได้ว่า: {pred}")
    print("รูปตัวอย่างที่ใกล้เคียงที่สุด 5 รูป (5 Nearest Neighbors):")
    for d, i in zip(dist[0], idx[0]):
        row = X_train.index[i]
        n_diff = round(d * len(FEATURES))           # แปลงเป็นจำนวน feature ที่ต่างกัน
        print(f"  {df.loc[row, 'รูป']:<10} {y_train.loc[row]:<18} ต่างกัน {n_diff} ข้อ")
    print()
    return pred


# ---------------- ตัวอย่างการใช้งาน ----------------
print("=== ตัวอย่างการจำแนกรูปใหม่ ===")
predict_rash(["ตุ่มน้ำบนฐานสีแดง", "ตุ่มน้ำขุ่นหรือเป็นหนอง", "ผื่นหลายระยะในรูปเดียว"])
predict_rash(["ตุ่ม/ผื่นที่ฝ่ามือหรือฝ่าเท้า", "ตุ่มน้ำรูปวงรี สีขาวเทา มีขอบแดง"])
predict_rash(["ตุ่มเล็กมาก", "ตุ่มขึ้นหนาแน่นชิดกันเป็นปื้น"])
