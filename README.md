# Capstone Project: การทำนายความตั้งใจในการซื้อสินค้าของผู้เข้าชมเว็บไซต์ E-Commerce

## 1. ภาพรวมโครงงาน (Project Overview)

โครงงานนี้เป็นการประยุกต์ใช้เทคนิค **Data Mining และ Machine Learning ประเภท Classification** เพื่อวิเคราะห์และทำนายว่าผู้เข้าชมเว็บไซต์ E-Commerce จะทำการซื้อสินค้าภายใน Session นั้นหรือไม่ โดยพิจารณาจากพฤติกรรมการเข้าชมเว็บไซต์ เช่น จำนวนหน้าที่เข้าชม ระยะเวลาที่ใช้ในแต่ละประเภทของหน้าเว็บไซต์ อัตราการออกจากเว็บไซต์ ประเภทผู้เข้าชม และข้อมูลแวดล้อมของการเข้าชม

ผลลัพธ์ที่ต้องการคือโมเดลที่สามารถจำแนก Session ออกเป็น 2 กลุ่ม ได้แก่

- **Revenue = 1 (True):** Session ที่มีการซื้อสินค้า
- **Revenue = 0 (False):** Session ที่ไม่มีการซื้อสินค้า

โครงงานจะเปรียบเทียบประสิทธิภาพของ Classification Algorithms หลายรูปแบบ และวิเคราะห์ว่าปัจจัยใดมีความสำคัญต่อการทำนาย รวมถึงสรุปข้อค้นพบที่สามารถนำไปใช้ประกอบการทำความเข้าใจพฤติกรรมของผู้เข้าชมเว็บไซต์ได้

## 2. วัตถุประสงค์ (Objectives)

1. ศึกษาและทำความเข้าใจข้อมูลพฤติกรรมการเข้าชมเว็บไซต์ E-Commerce
2. สำรวจและวิเคราะห์ข้อมูลด้วย Exploratory Data Analysis (EDA)
3. เตรียมข้อมูลให้เหมาะสมสำหรับการสร้างโมเดล Classification
4. สร้างและเปรียบเทียบโมเดล Machine Learning เพื่อทำนายการซื้อสินค้า
5. ประเมินประสิทธิภาพของโมเดลและวิเคราะห์ข้อดีข้อจำกัดของแต่ละโมเดล
6. วิเคราะห์ปัจจัยสำคัญและสรุป Insight จากข้อมูล

## 3. แหล่งที่มาของข้อมูล (Dataset Source)

โครงงานใช้ Dataset ชื่อ **Online Shoppers Purchasing Intention Dataset** ซึ่งดาวน์โหลดจาก Kaggle

- **Kaggle Dataset:** [Online Shoppers Purchasing Intention Dataset](https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset)
- **ชื่อไฟล์ข้อมูล:** `online_shoppers_intention.csv`
- **ผู้เผยแพร่ชุดข้อมูลบน Kaggle:** Akash Patel
- **แหล่งข้อมูลต้นฉบับ:** UCI Machine Learning Repository
- **หน้า Dataset ต้นฉบับของ UCI:** [Online Shoppers Purchasing Intention Dataset – UCI](https://archive.ics.uci.edu/dataset/468/online)
- **DOI:** [https://doi.org/10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q)

หมายเหตุ: Kaggle เป็นแหล่งที่ดาวน์โหลด Dataset มาใช้ในโครงงาน ส่วนแหล่งข้อมูลต้นฉบับที่ระบุไว้ในรายละเอียดของชุดข้อมูลคือ UCI Machine Learning Repository

## 4. รายละเอียดของข้อมูล (Dataset Description)

Dataset ประกอบด้วยข้อมูลระดับการเข้าชมเว็บไซต์ (Session) จำนวน **12,330 Sessions** จากช่วงระยะเวลาประมาณ 1 ปี โดยแต่ละแถวแทนข้อมูลพฤติกรรมการเข้าชมหนึ่ง Session

ข้อมูลมีทั้งหมด 18 Columns แบ่งเป็น

- **Input Features:** 17 ตัวแปร ประกอบด้วย Numerical Features 10 ตัว และ Categorical Features 7 ตัว
- **Target Variable:** 1 ตัวแปร คือ `Revenue`

### 4.1 รายละเอียดตัวแปร

| ลำดับ | Feature | รายละเอียด |
|---:|---|---|
| 1 | `Administrative` | จำนวนหน้าเว็บไซต์หมวด Administrative ที่ผู้เข้าชมเปิดดู |
| 2 | `Administrative_Duration` | ระยะเวลารวมที่ใช้ในการดูหน้า Administrative (วินาที) |
| 3 | `Informational` | จำนวนหน้าเว็บไซต์หมวด Informational ที่ผู้เข้าชมเปิดดู |
| 4 | `Informational_Duration` | ระยะเวลารวมที่ใช้ในการดูหน้า Informational (วินาที) |
| 5 | `ProductRelated` | จำนวนหน้าเว็บไซต์ที่เกี่ยวข้องกับสินค้าที่ผู้เข้าชมเปิดดู |
| 6 | `ProductRelated_Duration` | ระยะเวลารวมที่ใช้ในการดูหน้าสินค้า (วินาที) |
| 7 | `BounceRates` | อัตราการเข้าชมหน้าเว็บไซต์แล้วออกโดยไม่มีการโต้ตอบเพิ่มเติม |
| 8 | `ExitRates` | อัตราการออกจากเว็บไซต์จากหน้าเว็บนั้นเมื่อเทียบกับจำนวนการดูหน้านั้น |
| 9 | `PageValues` | ค่าประมาณมูลค่าของหน้าเว็บที่ผู้เข้าชมเปิดดูก่อนธุรกรรม E-Commerce |
| 10 | `SpecialDay` | ระดับความใกล้เคียงของวันที่เข้าชมกับวันสำคัญหรือเทศกาล มีค่าระหว่าง 0 ถึง 1 |
| 11 | `Month` | เดือนที่เกิดการเข้าชมเว็บไซต์ |
| 12 | `OperatingSystems` | รหัสระบบปฏิบัติการที่ผู้เข้าชมใช้งาน |
| 13 | `Browser` | รหัสเว็บเบราว์เซอร์ที่ผู้เข้าชมใช้งาน |
| 14 | `Region` | รหัสภูมิภาคของผู้เข้าชม |
| 15 | `TrafficType` | รหัสประเภทแหล่งที่มาของ Traffic ที่นำผู้เข้าชมมายังเว็บไซต์ |
| 16 | `VisitorType` | ประเภทผู้เข้าชม เช่น New Visitor หรือ Returning Visitor |
| 17 | `Weekend` | ระบุว่าการเข้าชมเกิดขึ้นในวันเสาร์หรือวันอาทิตย์หรือไม่ |
| 18 | `Revenue` | ตัวแปรเป้าหมาย ระบุว่า Session นั้นมีการซื้อสินค้าหรือไม่ |

### 4.2 Target Variable

ตัวแปรเป้าหมายคือ `Revenue` ซึ่งใช้สำหรับงาน Binary Classification

| ค่า | ความหมาย |
|---|---|
| `False` หรือ `0` | ไม่มีการซื้อสินค้า (Not Purchased) |
| `True` หรือ `1` | มีการซื้อสินค้า (Purchased) |

### 4.3 การกระจายตัวของ Target หลังทำความสะอาด

หลังจากตรวจสอบและลบแถวข้อมูลซ้ำแบบตรงกันทุกคอลัมน์ (Exact Duplicate Rows) จำนวน 125 แถว ข้อมูลเหลือ 12,205 Sessions โดยมีการกระจายตัวของ Target ดังนี้

| กลุ่ม | จำนวน Sessions | สัดส่วนโดยประมาณ |
|---|---:|---:|
| ไม่ซื้อสินค้า (`Revenue = 0`) | 10,297 | 84.37% |
| ซื้อสินค้า (`Revenue = 1`) | 1,908 | 15.63% |
| **รวม** | **12,205** | **100%** |

ข้อมูลจึงมีลักษณะ **Class Imbalance** เนื่องจากจำนวน Session ที่ไม่ซื้อสินค้ามีมากกว่ากลุ่มที่ซื้อสินค้าอย่างชัดเจน การประเมินโมเดลจึงไม่ควรพิจารณาเฉพาะ Accuracy แต่ควรดู Precision, Recall และ F1-score ของกลุ่มผู้ซื้อร่วมด้วย

## 5. การเตรียมและสำรวจข้อมูล (Data Preparation & EDA)

ขั้นตอนการเตรียมข้อมูลและสำรวจข้อมูลที่ดำเนินการในโครงงานประกอบด้วย

1. โหลด Dataset และตรวจสอบโครงสร้างข้อมูล จำนวนแถว/คอลัมน์ และชนิดข้อมูล
2. ตรวจสอบ Missing Values และค่าติดลบในตัวแปรเชิงตัวเลข
3. ตรวจสอบข้อมูลซ้ำและลบแถวที่ซ้ำกันทุกคอลัมน์จำนวน 125 แถว
4. ตรวจสอบ Outliers ด้วยวิธี Interquartile Range (IQR) โดยเก็บค่าที่ตรวจพบไว้ เนื่องจากยังไม่มีหลักฐานว่าเป็นข้อมูลผิดพลาด และอาจเป็นพฤติกรรมการเข้าชมที่เกิดขึ้นจริง
5. สำรวจการกระจายตัวของ `Revenue` เพื่อทำความเข้าใจปัญหา Class Imbalance
6. วิเคราะห์จำนวนและ Purchase Rate แยกตาม `VisitorType`
7. วิเคราะห์ Purchase Rate ตามเดือน (`Month`)
8. วิเคราะห์การกระจายตัวของระยะเวลาการเข้าชมหน้าสินค้า (`ProductRelated_Duration`)
9. วิเคราะห์ Correlation ระหว่างตัวแปรเชิงตัวเลขและ `Revenue`

## 6. การเตรียมข้อมูลสำหรับ Machine Learning

- กำหนด `Revenue` เป็น Target Variable และใช้ตัวแปรอื่นเป็น Input Features
- แปลง `Revenue` เป็นค่า 0 และ 1 สำหรับการสร้างโมเดล
- แบ่งข้อมูลเป็น Training Set 80% และ Testing Set 20%
- ใช้ `stratify=y` เพื่อรักษาสัดส่วนของ Target ใน Training Set และ Testing Set
- กำหนด `random_state=42` เพื่อให้การแบ่งข้อมูลสามารถทำซ้ำได้
- ข้อมูลหลังทำความสะอาดมี 12,205 แถว แบ่งเป็น Training Set 9,764 แถว และ Testing Set 2,441 แถว
- ใช้ One-Hot Encoding กับ Categorical Features สำหรับโมเดลที่ไม่รองรับข้อมูล Categorical โดยตรง
- สำหรับ Logistic Regression มีการใช้ RobustScaler กับ Numerical Features และ One-Hot Encoding กับ Categorical Features
- สำหรับ CatBoost ส่งรายชื่อตัวแปร Categorical ให้โมเดลโดยตรง
- สำหรับ Random Forest, XGBoost และ Decision Tree ใช้ข้อมูลที่ผ่าน One-Hot Encoding โดยไม่ทำ Scaling

## 7. โมเดล Classification ที่ใช้

โครงงานทดลองเปรียบเทียบโมเดลทั้งหมด 5 Algorithms ได้แก่

| Model | แนวคิดโดยสรุป |
|---|---|
| Logistic Regression | โมเดล Classification พื้นฐานที่ใช้ความสัมพันธ์ของตัวแปรเพื่อประมาณโอกาสการเกิด Class |
| Decision Tree | โมเดลที่แบ่งข้อมูลตามเงื่อนไขของ Features เป็นโครงสร้างต้นไม้ |
| Random Forest | โมเดล Ensemble ที่รวมผลจาก Decision Trees หลายต้น |
| XGBoost | โมเดล Gradient Boosting ที่สร้างต้นไม้ต่อเนื่องเพื่อปรับปรุงข้อผิดพลาด |
| CatBoost | โมเดล Gradient Boosting ที่รองรับ Categorical Features โดยตรง |

## 8. การประเมินประสิทธิภาพโมเดล (Model Evaluation)

ใช้ชุด Testing Set เดียวกันในการประเมินโมเดล โดยพิจารณาค่าเหล่านี้

- **Accuracy:** สัดส่วนของการทำนายที่ถูกต้องจากข้อมูลทั้งหมด
- **Precision:** ในกลุ่มที่โมเดลทำนายว่าเป็นผู้ซื้อ มีสัดส่วนเท่าใดที่เป็นผู้ซื้อจริง
- **Recall:** จากผู้ซื้อจริงทั้งหมด โมเดลสามารถตรวจจับได้กี่ส่วน
- **F1-score:** ค่าที่สรุปความสมดุลระหว่าง Precision และ Recall
- **Confusion Matrix:** ตารางแสดงจำนวนผลทำนายที่ถูกและผิดในแต่ละ Class

เนื่องจากข้อมูลมี Class Imbalance การพิจารณาโมเดลที่เหมาะสมจะให้ความสำคัญกับผลของ Class 1 (Purchased) ร่วมกับ Accuracy ไม่เลือกโมเดลจาก Accuracy เพียงค่าเดียว

## 9. ไฟล์ที่เกี่ยวข้องกับโครงงาน

- `Project_DataMining.ipynb` — Notebook สำหรับการโหลดข้อมูล สำรวจข้อมูล เตรียมข้อมูล สร้างโมเดล และประเมินผล
- `online_shoppers_intention.csv` — ไฟล์ Dataset ที่ดาวน์โหลดจาก Kaggle
- `README.md` — เอกสารอธิบายรายละเอียดโครงงานและที่มาของ Dataset

## 10. แหล่งอ้างอิง (References)

1. Akash Patel. *Online Shoppers Purchasing Intention Dataset*. Kaggle. [https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset](https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset)
2. Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. [https://doi.org/10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q)
3. Sakar, C. O., Polat, S. O., Katircioglu, M., & Kastro, Y. (2019). *Real-time prediction of online shoppers’ purchasing intention using multilayer perceptron and LSTM recurrent neural networks*. Neural Computing and Applications.

---

> **หมายเหตุ:** ผลการทดลองและการเลือกโมเดลในเอกสารนี้เป็นส่วนหนึ่งของการดำเนินโครงงาน และอาจมีการเปลี่ยนแปลงเมื่อมีการปรับแต่งโมเดลหรือทำการทดลองเพิ่มเติม
