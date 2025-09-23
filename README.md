# PDPA Assistant with LangGraph

ระบบช่วยเหลือเกี่ยวกับ พ.ร.บ. คุ้มครองข้อมูลส่วนบุคคล (PDPA) ที่ใช้ LangGraph และ Ollama

## การติดตั้ง

1. ติดตั้ง dependencies:
```bash
pip install -r requirements.txt
```

2. ติดตั้ง Tesseract OCR:
   - Windows: ดาวน์โหลดจาก https://github.com/UB-Mannheim/tesseract/wiki
   - macOS: `brew install tesseract`
   - Linux: `sudo apt-get install tesseract-ocr`

3. ตั้งค่า Ollama:
```bash
# ติดตั้ง Ollama จาก https://ollama.ai/
# รัน model ที่ต้องการ
ollama pull hf.co/Float16-cloud/typhoon2-qwen2.5-7b-instruct-gguf:Q8_0
```

4. ตั้งค่า API Keys (ไม่บังคับ):
   - สร้างไฟล์ `.env` ในโฟลเดอร์หลัก
   - เพิ่ม SERPER_API_KEY สำหรับ web search:
```
SERPER_API_KEY=your_serper_api_key_here
```

## การใช้งาน

รันแอปพลิเคชัน:
```bash
streamlit run app_llama3.2.py
```

## คุณสมบัติ

- 🔍 ค้นหาข้อมูลจาก PDF และฐานความรู้
- 🌐 ค้นหาข้อมูลเพิ่มเติมจากเว็บ (ต้องมี SERPER_API_KEY)
- 🤖 ใช้ LangGraph workflow สำหรับการประมวลผล
- 📊 สร้างคำตอบหลายแบบและจัดอันดับ
- 🎨 UI ที่สวยงามและใช้งานง่าย

## การตั้งค่า Web Search

เพื่อใช้ web search:
1. ไปที่ https://serper.dev/
2. สร้างบัญชีและรับ API key
3. เพิ่ม API key ในไฟล์ `.env`:
```
SERPER_API_KEY=your_actual_api_key_here
```
4. ติดตั้ง serper-dev:
```bash
pip install serper-dev
```

## โครงสร้างโปรเจค

```
agentic_rag/
├── app_llama3.2.py          # Streamlit app หลัก
├── src/agentic_rag/
│   ├── crew.py              # LangGraph workflow
│   ├── tools/               # Custom tools
│   └── config/              # Configuration files
├── knowledge/               # ฐานความรู้ PDPA
└── assets/                  # รูปภาพและไฟล์อื่นๆ
``` 