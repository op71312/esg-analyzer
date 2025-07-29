# ESG News Analyzer & RAG Chatbot for Thai Stocks
---

## โปรเจกต์นี้คืออะไร?

โปรเจกต์นี้นำเสนอ **ระบบวิเคราะห์ข่าวหุ้นไทยเชิง ESG (Environmental, Social, Governance)** ด้วยโมเดล LLM ที่ถูก fine-tune เฉพาะทาง และระบบ RAG (Retrieval-Augmented Generation) เพื่อให้วิเคราะห์ข่าวได้อย่างแม่นยำและอ้างอิงข้อมูลจริงจากเอกสาร ESG ของตลาดหลักทรัพย์ไทย

- **สร้างชุดข้อมูลข่าวหุ้นไทย** พร้อมวิเคราะห์ Sentiment และผลกระทบต่อ ESG Rating
- **Fine-tune LLM (Llama 3.2B)** ด้วยข้อมูลข่าวที่เตรียมเอง
- **Deploy โมเดลผ่าน Ollama** เพื่อใช้งานกับ LangChain
- **RAG ด้วย WangchanBERTa** สำหรับฝังข่าวและเอกสาร ESG ลง Vector Database (FAISS)
- **Chatbot วิเคราะห์ข่าว** ที่อ้างอิงข้อมูลจากเอกสารจริง

---

## วัตถุประสงค์

1. **สร้างโมเดลวิเคราะห์ข่าวหุ้นไทยเชิง ESG** ที่เข้าใจบริบทและให้เหตุผลได้
2. **เชื่อมโยงข่าวกับข้อมูล ESG จริง** ด้วย RAG เพื่อเพิ่มความแม่นยำ
3. **เปิดให้ใช้งานผ่าน LangChain** สำหรับงานวิเคราะห์ข่าวหุ้นอัตโนมัติ

---

## ขั้นตอนการทำ

### 1. การเตรียมข้อมูล

- **ค้นหาข่าวหุ้นด้วย ChatGPT**  
  ใช้ prompt:
  ```
  ฉันอยากให้คุณหาข่าวล่าสุดของหุ้น 5 ตัวนี้ 
  TTCL
  TTW
  TU
  TVDH
  TVO 
  อย่างละ 5 ข่าว โดยมีข่าวปีล่าสุด 3 ข่าว และปีก่อน ๆ 2 ข่าว
  โดยประกอบไปด้วย 
  1.หัวข้อข่าว 
  2.วันที่เขียนข่าว แสดงผลแบบ วว/ดด/ปปปป
  3.เนื้อหาข่าว
  4.วิเคราะห์ด้วยว่าข่าวที่คุณให้มาส่งผลต่อหุ้นนั้นในทางบวก กลาง หรือลบ
  5.เหตุผล
  ```
  - เปลี่ยนชื่อหุ้นวนไปจนครบทุกตัวที่ต้องการ
  - สุ่มตรวจสอบแหล่งข่าวจริงเป็นระยะ

- **จัดรูปแบบข้อมูล**
  - นำข้อมูลใส่ Word แล้วแทนที่อักขระพิเศษด้วยช่องว่าง
  - แปลงเป็น plain text (UTF-8)
  - นำเข้า ChatGPT เพื่อแปลงเป็น CSV

### 2. Clean Data

- ใช้ `cleandata.ipynb`  
  - ล้างข้อมูล, จัดกลุ่ม sentiment, ตรวจสอบ missing values  
  - บันทึกเป็น news.csv สำหรับใช้เทรนโมเดล

### 3. Fine-tune LLM (Llama 3.2B) ด้วย Unsloth

- ใช้ finetune_llama3_2unsloth_financeipynb.ipynb
  - Merge ข้อมูลข่าวกับ ESG Ratings
  - สร้าง prompt-response สำหรับ fine-tune
  - เทรนด้วย Unsloth/LoRA
  - Export เป็นไฟล์ `.gguf` (เช่น unsloth.Q4_K_M.gguf)

### 4. Deploy โมเดลผ่าน Ollama

- วางไฟล์ unsloth.Q4_K_M.gguf และ `Modelfile` ในโฟลเดอร์เดียวกัน
- สร้างโมเดล Ollama ด้วยคำสั่ง:
  ```sh
  ollama create esg-analyzer -f Modelfile
  ```

### 5. ทดสอบโมเดล LLM

- ใช้ chatbot.ipynb  
  - ทดสอบวิเคราะห์ข่าวหุ้นด้วย prompt template  
  - ตรวจสอบความแม่นยำและเหตุผลที่โมเดลให้

### 6. สร้าง RAG ด้วย WangchanBERTa + FAISS

- ใช้ rag.ipynb
  - อ่านเอกสาร ESG PDF
  - ตัดประโยค, สร้าง embedding ด้วย `"airesearch/wangchanberta-base-att-spm-uncased"`
  - เก็บลง FAISS Vector Database (chunk_faiss.index, chunk_texts.pkl)

### 7. ทดสอบ RAG Chatbot

- ใช้ chatwithrag.ipynb
  - ดึง context จากเอกสาร ESG ด้วย embedding
  - ส่ง context + ข่าวเข้า LLM ผ่าน LangChain
  - ได้ผลลัพธ์ที่อ้างอิงข้อมูลจริงจากเอกสาร

---

## Dependencies

```text
transformers
torch
faiss
langchain
ollama
pypdf
pythainlp
pandas
numpy
unsloth
```

---

## ตัวอย่างการใช้งาน

1. เตรียมข้อมูลข่าวและ ESG ตามขั้นตอนข้างต้น
2. เทรนและ deploy โมเดล LLM ด้วย Ollama
3. สร้าง FAISS index จากเอกสาร ESG
4. รัน chatwithrag.ipynb เพื่อวิเคราะห์ข่าวหุ้นไทยเชิง ESG พร้อมอ้างอิง context จากเอกสารจริง

---

## หมายเหตุ

- ข้อมูลข่าวและ ESG ที่ใช้เป็นตัวอย่าง อาจต้องอัปเดตตามรอบเวลา
- โมเดล LLM ที่ fine-tune แล้วเหมาะกับภาษาไทยและข่าวหุ้นไทยโดยเฉพาะ
- การใช้งาน RAG ช่วยให้โมเดลอ้างอิงข้อมูลจริงและลด Hallucination

---

## โครงสร้างไฟล์ที่สำคัญ

- news.csv — ข้อมูลข่าวหุ้นที่เตรียมเอง
- finetune_llama3_2unsloth_financeipynb.ipynb — โค้ดเทรน LLM
- unsloth.Q4_K_M.gguf, `Modelfile` — ไฟล์โมเดลสำหรับ Ollama
- rag.ipynb — โค้ดสร้าง FAISS index จากเอกสาร ESG
- chatwithrag.ipynb — ตัวอย่าง RAG Chatbot วิเคราะห์ข่าว

---
