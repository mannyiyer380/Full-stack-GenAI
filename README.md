# Full-Stack Generative AI Bootcamp — GenAI Workspace
This repository contains my hands-on coursework, notes, and project artifacts for the **Full-Stack Generative AI Bootcamp v1.0**.

## Program overview
- **Track:** Full-Stack Generative AI Bootcamp (v1.0)
- **Focus areas:** LLM foundations, fine-tuning, RAG, agentic systems, evaluation, guardrails, MCP, and cloud deployment
- **Estimated pace:** ~5–6 months at ~6 hours/week
- **Recommended background:** Python, APIs (REST/JSON), Git/CLI, basic ML concepts

## Learning objectives
- Understand LLM foundations (transformers, tokenization, embeddings)
- Build and integrate LLM APIs with cost/latency awareness
- Apply fine-tuning approaches (including PEFT, LoRA, QLoRA)
- Design and implement production-grade RAG systems
- Build single-agent and multi-agent workflows
- Evaluate GenAI systems for quality, reliability, and cost
- Implement safety guardrails and prompt injection defenses
- Deploy scalable GenAI systems on cloud platforms

## Curriculum map (from syllabus)
1. Foundations of Modern GenAI  
2. Understanding LLMs, SLMs & Multimodal LLMs  
3. API for Accessing LLMs  
4. Fine-Tuning Techniques  
5. LLM Hosting on Your Own Server and Exposing as an API  
6. Prompt Engineering  
7. Retrieval-Augmented Generation (RAG) Systems  
8. Advanced RAG & Multimodal Systems  
9. Agents, Multi-Agent & Deep Agent Systems  
10. Evaluation Strategies  
11. Guardrails  
12. MCP  
13. Cloud Services for GenAI (AWS)  
14. No-Code Agent Tools  
15. Claude Automation Mastery  
16. End-to-End Project: Enterprise Document Intelligence System  
17. End-to-End Project: AI-Powered Report Automation System  
18. End-to-End Project: AI-Powered Software Lifecycle Automation System

## Repository layout
This workspace is organized by study date and module work.

```text
GenAI/
├── 29_Mar_2026/                # Early module notebooks and requirements
├── 05_Apr_26/                  # NLP feature engineering assignment + notebook
├── 11_Apr_2026/                # Newer session workspace
├── Full-Stack GenAI Bootcamp v1.0.pdf
├── sum.py
└── README.md
```

## Working with this repository
1. Open the folder for the specific session/module.
2. Create a virtual environment (if not already created).
3. Install dependencies from that folder’s `requirements.txt`.
4. Run notebooks or scripts for that module.

Example:

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Notes
- The syllabus reference used for this README is: `Full-Stack GenAI Bootcamp v1.0.pdf`.
- A repository-level `.gitignore` is included to avoid committing local environments, cache files, and OS artifacts.
