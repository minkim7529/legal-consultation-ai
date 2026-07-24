# 법률상담 AI — 자료

Llama 3.1 8B를 LoRA로 파인튜닝한 5인 팀 졸업작품의 실제 학습 코드와 보고서입니다.

## 파일

| 파일 | 내용 |
|---|---|
| [`법률_데이터셋_학습.ipynb`](법률_데이터셋_학습.ipynb) | 데이터셋 정제 + LoRA/SFTTrainer 학습 코드 (Google Colab) |
| [`법률상담사이트_최종.ipynb`](법률상담사이트_최종.ipynb) | Gradio 기반 법률 상담 웹 인터페이스 코드 |
| [`졸업작품 최종 보고서.hwp`](졸업작품%20최종%20보고서.hwp) | 졸업작품 최종 보고서 |
| [`법률상담 ai 최종보고서.pptx`](법률상담%20ai%20최종보고서.pptx) | 발표자료 (공학제 제출용) |
| [`법률상담 ai 논문 초안.pdf`](법률상담%20ai%20논문%20초안.pdf) | 논문 초안 — **학술지 게재 여부는 확인되지 않음**, 작성한 초안입니다 |
| [`사용설명서.txt`](사용설명서.txt) | 학습 → 배포까지 실행 순서 |
| [`dataset_sample_40rows.jsonl`](dataset_sample_40rows.jsonl) | 학습 데이터셋 샘플 40행 (전체 56,358행 중 일부) |
| [`model-adapter-config/`](model-adapter-config) | 학습된 LoRA 어댑터의 설정 파일 |

## 용량 문제로 전체를 올리지 않은 것

- **학습 데이터셋 전체** (`reformatted_law_dataset_cleaned_final.jsonl`, 약 75MB,
  56,358행) — 위 40행 샘플로 대체. 원본은 Hugging Face에 공개된 한국어 법률
  질문-답변 코퍼스를 정제한 것입니다.
- **학습된 LoRA 가중치** (`adapter_model.safetensors`, 약 27MB)와
  **토크나이저 전체 파일**(`tokenizer.json`, 약 16MB) — 대신
  `model-adapter-config/`에 `adapter_config.json`(LoRA rank/alpha 등 하이퍼파라미터),
  `tokenizer_config.json`, `special_tokens_map.json`만 포함했습니다. 실제
  가중치 파일이 필요하면 원본은 로컬(`unsloth_law_model_lora.zip`)에 보관되어
  있습니다.

## 공동 저자

김지환, 김두한, 김민석, 신영훈, 김희철 — 대구대학교 5인 팀 프로젝트.
