AI Cancer Screening Agent

An AI-powered cancer risk screening assistant designed to provide early-stage risk insights, educational guidance, and personalized recommendations based on user inputs.
The system combines Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) to deliver context-aware and reliable responses grounded in medical knowledge.

Overview

This project focuses on building a practical, user-facing healthcare AI system that bridges machine learning with real-world deployment. It enables users to input symptoms and relevant personal information, and receive structured, non-diagnostic insights to support early awareness.
The application has been deployed in an academic environment and used by approximately 100 users.
Features
Risk assessment based on symptoms, lifestyle factors, and demographic inputs
Context-aware reasoning using Large Language Models
Retrieval-Augmented Generation (RAG) with ChromaDB for grounded responses
Real-time API built with FastAPI
Frontend developed using Lovable for rapid prototyping and usability
End-to-end system from user interaction to AI-generated recommendations
Tech Stack

Backend:

Python
FastAPI

Frontend:

Lovable (AI-assisted UI development)

AI/ML:

Large Language Models (LLMs)
LangChain

Data Layer:

ChromaDB (vector database for RAG)

Deployment:

University-hosted server
System Workflow
User inputs symptoms and relevant personal information
Input data is structured and preprocessed
Relevant medical knowledge is retrieved from the vector database
The LLM performs contextual reasoning using both input and retrieved data
The system generates:
Risk insights (non-diagnostic)
Educational explanations
Preventive recommendations
Disclaimer

This system is not a medical diagnostic tool. It is intended for educational and early awareness purposes only. Users should consult licensed medical professionals for any clinical decisions or concerns.

Future Improvements
Integration with clinical-grade datasets
Confidence scoring and uncertainty estimation
Multi-language support
Longitudinal user health tracking
Improved explainability of model outputs

Author
Developed as part of graduate-level work in Engineering Data Science, with a focus on applied AI, healthcare systems, and real-world deployment.

