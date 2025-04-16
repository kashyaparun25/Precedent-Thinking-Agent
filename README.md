# Precedents Thinking System

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.0+-red.svg)

A research-intensive system built with Streamlit and CrewAI to discover cross-industry precedents, synthesize innovative solutions, and generate comprehensive reports for complex challenges.

## Overview

The Precedents Thinking System is designed to help users tackle complex challenges by:
1. Framing challenges with contextual document analysis
2. Searching for precedents across multiple domains (academic papers, news, web, industry cases)
3. Synthesizing precedents into innovative solutions
4. Visualizing relationships between precedents
5. Exporting results in various formats
6. Generating detailed research reports

## Features

- **Multi-source Search**: Integrates NewsAPI, Google Scholar, Serper API, and DuckDuckGo for comprehensive precedent discovery
- **Document Processing**: Supports PDF, DOCX, and TXT file analysis
- **AI-Powered Synthesis**: Uses CrewAI agents with Gemini LLM for intelligent analysis
- **Interactive Visualizations**: Includes mind maps and source distribution visuals
- **Export Capabilities**: Offers CSV, JSON, and Markdown report exports
- **Customizable Reports**: Generates structured research reports with actionable recommendations

## System Architecture

```mermaid
graph TD
    A[User Interface<br>Streamlit] --> B[Challenge Framing]
    A --> C[API Configuration]
    A --> D[File Upload]
    
    B --> E[Precedent Search]
    D --> E
    C --> E
    
    E -->|NewsAPI| F[News Search]
    E -->|Scholarly| G[Academic Search]
    E -->|Serper| H[Web Search]
    E -->|DuckDuckGo| I[Web Search]
    E -->|Document Processor| J[Document Analysis]
    
    E --> K[Idea Synthesis]
    K --> L[Research Report]
    K --> M[Visualization]
    
    L --> N[Export]
    M --> N
    
    subgraph CrewAI Agents
        E --> P[Precedents Explorer]
        K --> Q[Idea Synthesizer]
        L --> R[Report Generator]
        M --> S[Data Visualizer]
    end
    
    P -->|Gemini LLM| K
    Q -->|Gemini LLM| L
    R -->|Gemini LLM| N
    S -->|Gemini LLM| M
