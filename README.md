# AI_Powered_tutor

A project developed based on Intel's problem statements

Overview

The AI-Powered Tutor is an interactive learning platform that leverages artificial intelligence to provide personalized educational experiences. This web application offers courses in AI, machine learning, data science, and related fields, with features like assignments, quizzes, progress tracking, and an AI-powered assistant.

Features

User Authentication: Secure login and registration system
Course Management: Browse, enroll in, and complete courses
Interactive Learning: AI-powered chat assistant for course-related questions
Assignment System: Submit and track assignments with completion status
Progress Tracking: Monitor course completion and assignment progress
Dark/Light Mode: Theme toggle for user preference

Technologies Used

Frontend: HTML5, CSS3, JavaScript
Styling: Bootstrap 4.5
Security: SHA-512 password hashing with unique salts
Local Storage: For persistent user data and session management
AI Integration: Connection to backend AI service (Ollama) and a basic RAG model

Backend 

Python backend server (AI_Teacher.py for handling course related queries)
Ollama service running locally (AI_Classmate.py for serving user with general conversation)
Course PDFs in the course_data directory are used for training the RAG model

Usage

Register a new account or login with existing credentials
Browse available courses from the Courses page
Enroll in courses to access materials
Complete assignments and track progress
Use the AI tutor for course-related questions
And enjoy the learning experience with the unique and minimalist responsive design of this platform.
