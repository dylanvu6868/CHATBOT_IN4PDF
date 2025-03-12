# CHATBOT_IN4PDF

CHATBOT_IN4PDF is a project designed to enable AI-powered interaction with PDF documents, utilizing advanced language models for efficient data extraction and querying.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation Guide](#installation-guide)
- [Usage](#usage)
- [Features](#features)
- [Contribution](#contribution)
- [License](#license)

## Introduction

CHATBOT_IN4PDF allows users to interact with the content of PDF files through a chatbot interface. The project employs natural language processing (NLP) techniques and integrates AI models to extract, analyze, and answer queries based on the information contained in PDF documents.

## Project Structure

The project consists of the following directories and files:

- `data/`: Contains PDF documents and related datasets.
- `models/`: Pretrained and fine-tuned AI models for text processing.
- `src/`: Source code for chatbot logic and data handling.
- `notebooks/`: Jupyter notebooks for development and testing.
- `preparing_vector_db.py`: Script for preprocessing PDFs and generating vector embeddings.
- `qa_bot.py`: Main script to run the chatbot and handle user queries.
- `requirements.txt`: List of dependencies required to run the project.
- `README.md`: This documentation file.

## Model 1. https://huggingface.co/vilm/vinallama-7b-chat-GGUF/tree/main 
## Model 2. https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/blob/main/all-MiniLM-L6-v2.F16.gguf
## Installation Guide

To set up the project, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/dylanvu6868/CHATBOT_IN4PDF.git
   cd CHATBOT_IN4PDF
   ```
2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. Run the script to preprocess PDF files and generate vector embeddings:
   ```sh
   python preparing_vector_db.py
   ```
2. Start the chatbot to interact with your PDF data:
   ```sh
   python qa_bot.py
   ```
3. Enter queries in natural language, and the chatbot will provide answers based on the PDF content.

## Features

- AI-driven document analysis and retrieval
- Natural language interaction with PDF content
- Support for multiple PDF files
- Efficient vector search for relevant information
- Customizable AI models for improved responses

## Contribution

We welcome contributions! Follow these steps:

1. Fork the repository.
2. Create a new branch (`feature-branch`).
3. Make your changes and commit them.
4. Push to the branch and create a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
