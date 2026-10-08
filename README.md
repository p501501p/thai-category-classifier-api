# ระบบจำแนกหมวดหมู่ข้อความภาษาไทยด้วย ANN

## บทคัดย่อ

โครงงานนี้พัฒนาระบบสำหรับจำแนกข้อความภาษาไทยให้อยู่ในหนึ่งใน 6 หมวดหมู่ โดยแปลงข้อความเป็นคุณลักษณะด้วย TF-IDF ลดมิติข้อมูลด้วย Truncated SVD แล้วส่งเข้าโครงข่ายประสาทเทียมแบบหลายชั้น (Artificial Neural Network: ANN) ผลลัพธ์ประกอบด้วยหมวดหมู่ที่คาดการณ์และค่าความเชื่อมั่น ระบบให้บริการผ่าน REST API เพื่อให้ n8n หรือโปรแกรมอื่นส่งข้อความมาขอผลทำนายได้

## วัตถุประสงค์

- พัฒนาต้นแบบการจำแนกข้อความภาษาไทยตามเนื้อหา
- ทดลองใช้คุณลักษณะทั้งระดับคำและระดับตัวอักษรร่วมกัน
- ให้บริการโมเดลผ่าน API เพื่อเชื่อมต่อกับระบบอัตโนมัติ เช่น n8n

## ข้อมูลและหมวดหมู่

ชุดข้อมูลที่ใช้พัฒนาโมเดลอยู่ในไฟล์ `ANN_Train_Data_6Categories_800rows.csv` มีข้อมูล 800 แถว และมีคอลัมน์ `Text` (ข้อความ), `Category_ID` (รหัสหมวดหมู่), `Category` (ชื่อหมวดหมู่) และ `PostId` (รหัสโพสต์) ก่อนฝึกโมเดล โค้ดจะตัดแถวที่ไม่มีข้อความหรือรหัสหมวดหมู่ ตัดช่องว่างหัวท้าย แปลงรหัสหมวดหมู่เป็นจำนวนเต็ม และนำข้อความซ้ำออก

หมายเหตุ: repository นี้จัดเตรียมส่วน API สำหรับเรียกใช้โมเดล โดยไฟล์ชุดข้อมูลและสคริปต์ฝึกโมเดลเป็นไฟล์ต้นทางของขั้นตอนพัฒนา ไม่ได้รวมอยู่ในชุด deploy นี้

| รหัส | หมวดหมู่ |
|---:|---|
| 0 | การรับสมัครและประชาสัมพันธ์รับเข้า |
| 1 | การเรียนการสอนและวิชาการ |
| 2 | วิจัย นวัตกรรม และเทคโนโลยี |
| 3 | กิจกรรมนักศึกษา กีฬา และศิลปวัฒนธรรม |
| 4 | ข่าวสาร รางวัล และความร่วมมือ |
| 5 | ชุมชน สังคม และข้อมูลทั่วไป |

## วิธีการพัฒนาโมเดล

1. **แบ่งข้อมูล** — แบ่งข้อมูลเป็นชุดทดสอบ 20% และชุดข้อมูลที่เหลือ 80% จากนั้นแบ่งชุดฝึกและชุดตรวจสอบจากส่วน 80% ในอัตรา 85:15 ใช้ stratified split เพื่อรักษาสัดส่วนของแต่ละหมวดหมู่ และกำหนด random state เป็น 42
2. **แปลงข้อความเป็นคุณลักษณะ** — รวม TF-IDF สองชุด ได้แก่ TF-IDF ระดับคำแบบ unigram/bigram โดยใช้ PyThaiNLP แบ่งคำ และ TF-IDF ระดับตัวอักษร n-gram ขนาด 2–5 ตัว
3. **ลดมิติ** — ใช้ Truncated SVD ลดเวกเตอร์คุณลักษณะให้เหลือ 100 มิติ โดย fit ตัวแปลงจากชุดฝึกเท่านั้น แล้วนำตัวแปลงเดียวกันไปใช้กับชุดตรวจสอบและชุดทดสอบ
4. **ฝึก ANN** — โครงข่ายประกอบด้วยชั้น Dense ขนาด 64 และ 32 หน่วย ใช้ ReLU และ Dropout 0.4/0.2 ตามลำดับ ชั้นผลลัพธ์มี 6 หน่วยและใช้ Softmax ฝึกด้วย Adam, learning rate เริ่มต้น 0.0005, batch size 16 และไม่เกิน 150 epochs
5. **ควบคุมการฝึก** — ใช้ class weights เพื่อชดเชยจำนวนตัวอย่างแต่ละหมวดที่ไม่เท่ากัน ใช้ L2 regularization, Early Stopping จาก `val_loss` และลด learning rate เมื่อผลบนชุดตรวจสอบไม่ดีขึ้น
6. **ประเมินผล** — ประเมิน artifact โมเดลที่ใช้ให้บริการกับชุดทดสอบเดิม โดยทำความสะอาดข้อมูลและแบ่งชุดทดสอบแบบ stratified ด้วย `test_size=0.20` และ `random_state=42` ได้ผลทำนายถูก **104 จาก 146 ตัวอย่าง** คิดเป็น accuracy **71.23%**

ผลนี้คำนวณจากไฟล์โมเดลปัจจุบันใน `artifacts/` ไม่ใช่ค่าที่ดึงจาก log การฝึกเดิม และควรตีความภายใต้ชุดข้อมูลกับวิธีแบ่งข้อมูลที่ระบุไว้เท่านั้น

## การทำงานของระบบ

```text
ข้อความจากผู้ใช้หรือ n8n
        ↓
REST API ตรวจสอบข้อมูลและ API key
        ↓
TF-IDF (ระดับคำ + ระดับตัวอักษร)
        ↓
Truncated SVD (100 มิติ)
        ↓
ANN และ Softmax
        ↓
รหัสหมวดหมู่ + ชื่อหมวดหมู่ + confidence
```

ระหว่างการให้บริการ ระบบโหลด TF-IDF, SVD, น้ำหนัก ANN และข้อมูลชื่อหมวดหมู่จากไฟล์ใน `artifacts/` การคำนวณ ANN ใน API ใช้ NumPy กับน้ำหนักที่บันทึกไว้ เพื่อให้บริการทำนายโดยไม่ต้องฝึกโมเดลใหม่ทุกครั้ง

## เทคโนโลยีที่ใช้

- Python 3.12
- FastAPI และ Pydantic สำหรับ REST API และตรวจสอบ request
- PyThaiNLP สำหรับตัดคำภาษาไทย
- scikit-learn สำหรับ TF-IDF และ Truncated SVD
- TensorFlow/Keras สำหรับสร้างและฝึก ANN
- NumPy และ joblib สำหรับโหลด artifact และคำนวณผลทำนาย
- Vercel Functions สำหรับเผยแพร่ API
- n8n สำหรับเชื่อม API เข้ากับ workflow อัตโนมัติ

## API ที่ให้บริการ

URL สำหรับ Production: <https://thai-category-classifier-api.vercel.app>

### ตรวจสอบสถานะ

```http
GET /health
```

ตัวอย่างผลลัพธ์:

```json
{"status": "ok"}
```

### ทำนายหมวดหมู่

```http
POST /predict
Content-Type: application/json
X-API-Key: <API_KEY>
```

ตัวอย่าง request:

```json
{
  "text": "ประกาศรับสมัครนักศึกษาใหม่ ประจำปีการศึกษา"
}
```

ตัวอย่างรูปแบบ response (ค่าความเชื่อมั่นเป็นตัวอย่าง):

```json
{
  "category_id": 0,
  "category": "การรับสมัครและประชาสัมพันธ์รับเข้า",
  "confidence": 0.95
}
```

ค่า `confidence` เป็นค่าความน่าจะเป็นของหมวดหมู่ที่โมเดลเลือก ไม่ใช่การรับประกันว่าผลทำนายถูกต้อง API กำหนดความยาวข้อความตั้งแต่ 1 ถึง 30,000 ตัวอักษร และต้องส่ง API key ที่ถูกต้องใน header `X-API-Key`

### การตอบสนองเมื่อเกิดข้อผิดพลาด

- `401 Unauthorized` — ไม่มี API key หรือ API key ไม่ถูกต้อง
- `422 Unprocessable Entity` — request ไม่ตรงรูปแบบหรือข้อความว่าง
- `503 Service Unavailable` — ยังไม่ได้กำหนด `API_KEY` ใน environment
- `500 Internal Server Error` — เกิดข้อผิดพลาดระหว่างประมวลผล

## การเชื่อมต่อกับ n8n

สร้าง **HTTP Request** node และกำหนดค่า:

- Method: `POST`
- URL: `https://thai-category-classifier-api.vercel.app/predict`
- Body Content Type: `JSON`
- Body field: `text` เช่น `{{$json.Text}}`
- Header: `X-API-Key` โดยเก็บค่าเป็น credential ใน n8n

ห้ามใส่ API key ลงใน workflow ที่แชร์สาธารณะหรือ commit ลง Git

## การเผยแพร่ระบบ

ระบบนี้ deploy เป็น Python Function บน Vercel โดยใช้ `app.py` เป็น entry point และใช้ไฟล์โมเดลใน `artifacts/` ที่แนบไปกับ Function ผ่าน `vercel.json` ไม่ต้องรัน `NW_Model.py` หรือฝึกโมเดลใหม่ตอน deploy

### ทดลอง Deploy ด้วย ZIP ที่ดาวน์โหลดจาก GitHub

ขั้นตอนนี้ใช้กรณีดาวน์โหลด source code เป็น ZIP มาไว้ในเครื่อง แล้ว deploy โปรเจกต์ด้วย Vercel CLI โดยไม่ต้องเชื่อม Vercel กับ GitHub:

1. ในหน้า GitHub ของ repository เลือก **Code → Download ZIP** แล้วแตก ZIP ลงในเครื่อง
2. เปิดโฟลเดอร์ที่แตกไฟล์ ตรวจดูตำแหน่ง `app.py` และ `vercel.json` หากอยู่ในโฟลเดอร์ `Post` ให้เข้าไปใน `Post` ก่อน deploy; ถ้าไฟล์อยู่ที่โฟลเดอร์หลักที่แตก ZIP มา ก็ใช้โฟลเดอร์หลักนั้น
3. ตรวจให้มี `app.py`, `predict_api.py`, `text_utils.py`, `requirements.txt`, `vercel.json`, `.python-version` และ `artifacts/` ซึ่งอย่างน้อยต้องมี `tfidf_vectorizer.joblib`, `svd_components.npy`, `ann_weights.npz` และ `metadata.json`
4. ติดตั้ง Node.js หากเครื่องยังไม่มี เพราะ Vercel CLI ต้องใช้ npm จากนั้นเปิด PowerShell ในโฟลเดอร์โปรเจกต์ (โฟลเดอร์เดียวกับ `app.py` และ `vercel.json`) แล้วรัน:

```powershell
npm install --global vercel
vercel login
vercel
```

5. ทำตามคำถามของ CLI เพื่อสร้างหรือเชื่อม Vercel project และ deploy ครั้งแรกเป็น Preview
6. เปิดโปรเจกต์นั้นใน Vercel Dashboard ไปที่ **Settings → Environment Variables** แล้วเพิ่ม `API_KEY` พร้อม secret ของคุณ เลือก environment ที่จะใช้ (อย่างน้อย **Production**; เพิ่ม **Preview** ด้วยถ้าจะทดสอบ Preview)
7. กลับไปที่ PowerShell ในโฟลเดอร์เดิม แล้ว deploy Production:

```powershell
vercel --prod
```

หากแก้ไขหรือเพิ่ม Environment Variable หลัง deploy ต้อง deploy ใหม่อีกครั้งเพื่อให้ค่าใหม่มีผล ตัวแปรและไฟล์ `.env` ในเครื่องจะไม่ถูกส่งไปแทนการตั้งค่า Environment Variables บน Vercel

### ตรวจสอบหลัง Deploy

แทน `<DEPLOYMENT_URL>` ด้วย URL ที่ Vercel แสดง โดยไม่ต้องใส่ `/` ปิดท้าย:

1. เปิด `https://<DEPLOYMENT_URL>/health` ควรได้ `{"status":"ok"}`
2. ทดลอง `POST https://<DEPLOYMENT_URL>/predict` โดยส่ง JSON ที่มี `text` และ header `X-API-Key` ซึ่งต้องตรงกับค่า `API_KEY` ที่ตั้งใน Vercel ตัวอย่างทดสอบด้วย PowerShell:

```powershell
$url = "https://<DEPLOYMENT_URL>"
$headers = @{ "X-API-Key" = "<API_KEY>" }
$body = @{ text = "ประกาศรับสมัครนักศึกษาใหม่ ประจำปีการศึกษา" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$url/predict" -Headers $headers -ContentType "application/json" -Body $body
```

อย่าแชร์หรือ commit ค่า `<API_KEY>` จริง หาก `/health` ใช้งานได้แต่ `/predict` ตอบ `503` ให้ตรวจว่าได้ตั้ง `API_KEY` ใน Environment Variables ของ environment ที่ deploy แล้วและ deploy ใหม่หลังตั้งค่า หากได้ `500` ให้ตรวจ deployment logs และยืนยันว่าไฟล์ใน `artifacts/` อยู่ในโฟลเดอร์ที่ deploy

หลัง deploy ให้เปลี่ยน URL ใน HTTP Request node ของ n8n เป็น `https://<DEPLOYMENT_URL>/predict` และเก็บ API key เป็น credential แยกจาก workflow

การตั้งค่า region ของ Function อยู่ใน `vercel.json` (Singapore: `sin1`) ส่วน RAM ที่กำหนดได้ขึ้นอยู่กับแพ็กเกจ Vercel; แพ็กเกจ Hobby ใช้ค่าเริ่มต้นของ Vercel และปรับเองไม่ได้

## ผลตรวจสอบระบบเบื้องต้น

ทดสอบ API บน Production แล้ว โดย `/health` ตอบกลับ `200 OK`, `/predict` เมื่อส่ง API key ที่ถูกต้องตอบกลับ `200 OK` พร้อมผลทำนาย และ `/predict` เมื่อไม่ส่ง API key ตอบกลับ `401 Unauthorized` การทดสอบนี้ยืนยันการทำงานเบื้องต้นของ API ไม่ใช่ผลประเมินความแม่นยำของโมเดล

## โครงสร้างไฟล์สำคัญ

```text
.
├── app.py                 # FastAPI entry point
├── predict_api.py         # โหลด artifact และประมวลผลการทำนาย
├── text_utils.py          # ฟังก์ชันตัดคำภาษาไทย
├── requirements.txt       # Python dependencies
├── vercel.json            # ตั้งค่า region และแนบ model artifacts
└── artifacts/
    ├── ann_weights.npz
    ├── ann_model.keras
    ├── metadata.json
    ├── svd_components.npy
    └── tfidf_vectorizer.joblib
```

## ข้อจำกัดและแนวทางพัฒนาต่อ

- โมเดลเรียนรู้จากชุดข้อมูลที่ใช้ฝึกเท่านั้น คุณภาพและความครอบคลุมของข้อความในชุดข้อมูลจึงมีผลต่อผลทำนาย
- ค่า confidence เป็นผลจาก Softmax และไม่ควรตีความว่าเป็นค่าความแม่นยำที่ผ่านการปรับเทียบแล้ว
- ควรบันทึกผลการประเมิน เช่น accuracy, precision, recall และ F1-score แยกรายหมวดหมู่ พร้อม confusion matrix เพื่อรายงานผลอย่างครบถ้วน
- ควรทดสอบกับข้อมูลใหม่ที่ไม่ใช้ในการฝึก และปรับปรุงชุดข้อมูลเมื่อพบข้อความที่โมเดลจำแนกผิด
- ควรติดตามเวลาในการตอบสนองและหน่วยความจำเมื่อมีผู้ใช้งานจริง
