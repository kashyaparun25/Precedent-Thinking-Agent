__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from duckduckgo_search import DDGS
import pandas as pd
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime, timedelta
import json
import numpy as np
from typing import Dict, List, Type
from pydantic import BaseModel, Field
import requests
from urllib.parse import urlparse
import PyPDF2
from docx import Document
from io import BytesIO
from newsapi import NewsApiClient
from scholarly import scholarly
import os
import litellm

# Initialize session state
if 'precedents' not in st.session_state:
    st.session_state.precedents = []
if 'framed_challenge' not in st.session_state:
    st.session_state.framed_challenge = None
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []

# Configure litellm for debugging
#litellm._turn_on_debug()

# Model configuration based on environment
# Model configuration based on environment
def create_llm():
    if 'google_key' in st.session_state and st.session_state.google_key:
        return LLM(
            model="gemini/gemini-2.0-flash-thinking-exp",
            api_key=st.session_state.google_key  # Add this line
        )
    else:
        st.error("Google API key not configured. Please add your API key in the sidebar.")
        return None

# Then use this function when needed
gemini_llm = create_llm()

# Define Custom Tools for CrewAI
class NewsSearchToolInput(BaseModel):
    """Input schema for NewsSearchTool."""
    query: str = Field(..., description="Search query for news articles")

class NewsSearchTool(BaseTool):
    name: str = "NewsSearch"
    description: str = "Search for recent news articles on a specific topic"
    args_schema: Type[BaseModel] = NewsSearchToolInput
    
    def _run(self, query: str) -> str:
        """Execute the news search query and return results"""
        if not st.session_state.get('news_api_key'):
            return "News API key not configured. Please add your API key in the sidebar."
            
        newsapi = NewsApiClient(api_key=st.session_state.news_api_key)
        
        # Use a shorter time range (last 7 days instead of 30)
        from_date = (datetime.now() - timedelta(days=20)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            all_articles = newsapi.get_everything(
                q=query,
                from_param=from_date,
                to=to_date,
                language='en',
                sort_by='relevancy'
            )
            
            results = []
            if 'articles' in all_articles and all_articles['articles']:
                for article in all_articles['articles'][:5]:  # Limit to 5 articles
                    results.append({
                        "title": article.get('title', ''),
                        "url": article.get('url', ''),
                        "snippet": article.get('description', ''),
                        "source": article.get('source', {}).get('name', ''),
                        "published": article.get('publishedAt', ''),
                        "type": "news"
                    })
            return json.dumps(results, indent=2)
            
        except Exception as e:
            return f"Error performing news search: {str(e)}"

class ScholarSearchToolInput(BaseModel):
    """Input schema for ScholarSearchTool."""
    query: str = Field(..., description="Search query for academic papers")

class ScholarSearchTool(BaseTool):
    name: str = "ScholarSearch"
    description: str = "Search for academic papers and research on a specific topic"
    args_schema: Type[BaseModel] = ScholarSearchToolInput
    
    def _run(self, query: str) -> str:
        """Execute the scholar search query and return results"""
        try:
            search_query = scholarly.search_pubs(query)
            results = []
            
            # Add a check to handle the possibility of no results
            if search_query is None:
                return "No academic papers found for this query."
                
            for i in range(15):  # Get first 5 results
                try:
                    pub = next(search_query)
                    if pub and 'bib' in pub:  # Check if pub and bib exist
                        results.append({
                            "title": pub['bib'].get('title', ''),
                            "authors": pub['bib'].get('author', []),
                            "year": pub['bib'].get('year', ''),
                            "url": pub.get('pub_url', ''),
                            "citations": pub.get('num_citations', 0),
                            "type": "academic"
                        })
                except StopIteration:
                    break
                except (TypeError, KeyError):
                    continue  # Skip this result if it's malformed
            
            if not results:
                return "No valid academic papers found for this query."
                
            return json.dumps(results, indent=2)
            
        except Exception as e:
            return f"Error searching academic papers: {str(e)}"

class SerperSearchToolInput(BaseModel):
    """Input schema for SerperSearchTool."""
    query: str = Field(..., description="Search query for web content")

class SerperSearchTool(BaseTool):
    name: str = "SerperSearch"
    description: str = "Search the web using Serper API for comprehensive results"
    args_schema: Type[BaseModel] = SerperSearchToolInput
    
    def _run(self, query: str) -> str:
        """Execute the Serper search query and return results"""
        if not st.session_state.get('serper_key'):
            return "Serper API key not configured. Please add your API key in the sidebar."
            
        url = "https://google.serper.dev/search"
        payload = {
            "q": query,
            "k": st.session_state.serper_key
        }
        headers = {
            "X-API-KEY": st.session_state.serper_key,
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get('organic', []):
                results.append({
                    "title": item.get('title', ''),
                    "url": item.get('link', ''),
                    "snippet": item.get('snippet', ''),
                    "domain": urlparse(item.get('link', '')).netloc,
                    "type": "web"
                })
            
            if not results:
                return "No web results found for this query."
                
            return json.dumps(results[:5], indent=2)  # Limit to 5 results
            
        except Exception as e:
            return f"Error searching with Serper: {str(e)}"

class DuckDuckGoSearchToolInput(BaseModel):
    """Input schema for DuckDuckGoSearchTool."""
    query: str = Field(..., description="Search query for web content using DuckDuckGo")

class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGoSearch"
    description: str = "Search the web using DuckDuckGo for privacy-focused results"
    args_schema: Type[BaseModel] = DuckDuckGoSearchToolInput
    
    def _run(self, query: str) -> str:
        """Execute the DuckDuckGo search query and return results"""
        try:
            results = []
            ddgs = DDGS()
            for result in ddgs.text(keywords=query, max_results=5):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", ""),
                    "type": "web"
                })
            
            if not results:
                return "No DuckDuckGo results found for this query."
                
            return json.dumps(results, indent=2)
            
        except Exception as e:
            return f"Error searching with DuckDuckGo: {str(e)}"

class DocumentProcessorToolInput(BaseModel):
    """Input schema for DocumentProcessorTool."""
    document_text: str = Field(..., description="Text content from uploaded documents")

class DocumentProcessorTool(BaseTool):
    name: str = "DocumentProcessor"
    description: str = "Analyze document content to extract relevant information"
    args_schema: Type[BaseModel] = DocumentProcessorToolInput
    
    def _run(self, document_text: str) -> str:
        """Process the document text and extract key information"""
        if not document_text or len(document_text.strip()) < 10:
            return "No document content provided or document is too short to analyze."
            
        try:
            # Simple approach - return a summary of document length and content preview
            word_count = len(document_text.split())
            char_count = len(document_text)
            
            summary = f"""
            Document Analysis:
            - Word Count: {word_count}
            - Character Count: {char_count}
            - Content Preview: {document_text[:500]}...
            
            The document has been processed and is available for analysis.
            """
            
            return summary
            
        except Exception as e:
            return f"Error processing document: {str(e)}"

# CrewAI Agent Definitions
def create_precedents_explorer_agent():
    """Create and configure the Precedents Explorer agent"""
    return Agent(
        role="Precedents Explorer",
        goal="Discover and analyze relevant precedents from different domains for cross-industry innovation",
        backstory="""You are a skilled precedents researcher with expertise in identifying valuable examples, 
        case studies, and research across multiple domains. You excel at finding patterns and connections 
        between seemingly unrelated fields to inspire innovative solutions.""",
        verbose=True,
        allow_delegation=True,
        tools=[
            NewsSearchTool(),
            ScholarSearchTool(),
            SerperSearchTool(),
            DuckDuckGoSearchTool(),
            DocumentProcessorTool()
        ],
        llm=gemini_llm
    )

def create_synthesizer_agent():
    """Create and configure the Idea Synthesis agent"""
    return Agent(
        role="Idea Synthesizer",
        goal="Synthesize precedents into innovative solutions and cross-industry applications",
        backstory="""You are an expert in creative synthesis and innovation, able to recognize patterns
        across diverse fields and combine them into novel solutions. You excel at making connections 
        that others miss and finding practical applications for abstract concepts.""",
        verbose=True,
        allow_delegation=True,
        llm=gemini_llm
    )

def create_report_agent():
    """Create and configure the Report Generator agent"""
    return Agent(
        role="Research Report Generator",
        goal="Create comprehensive, well-structured reports that clearly communicate findings and recommendations",
        backstory="""You are a skilled report writer with expertise in presenting complex information 
        in clear, engaging ways. You excel at structuring information logically, highlighting key 
        insights, and providing actionable recommendations that drive innovation.""",
        verbose=True,
        allow_delegation=True,
        llm=gemini_llm
    )

def create_visualization_agent():
    """Create and configure the Visualization agent"""
    return Agent(
        role="Data Visualizer",
        goal="Create clear, informative visualizations that highlight patterns and relationships in precedents",
        backstory="""You are an expert in data visualization with a talent for presenting complex information
        in visually compelling ways. You excel at creating charts, networks, and other visualizations
        that reveal hidden patterns and make insights accessible.""",
        verbose=True,
        llm=gemini_llm
    )

# CrewAI Task Definitions
def create_precedent_search_task(agent, challenge_description):
    """Create a task for searching precedents"""
    return Task(
        description=f"""
        Search for relevant precedents for the following challenge:
        
        CHALLENGE:
        {challenge_description}
        
        Use multiple search tools to find precedents from different domains including:
        1. Academic research papers
        2. Recent news articles
        3. Industry case studies
        4. Web resources
        
        For each precedent, provide:
        - Title
        - Description
        - Domain/field
        - Type (academic, news, web, industry)
        - Relevance to the challenge
        - Source URL
        - Date (if available)
        
        Structure your response as a JSON array of precedent objects.
        """,
        expected_output="JSON array of precedent objects with all required fields",
        agent=agent
    )

def create_synthesis_task(agent, challenge_description, precedents_data):
    """Create a task for synthesizing precedents into insights"""
    return Task(
        description=f"""
        Synthesize the following precedents into innovative solutions for this challenge:
        
        CHALLENGE:
        {challenge_description}
        
        PRECEDENTS:
        {json.dumps(precedents_data, indent=2)[:3000]}  # Limit to prevent token overload
        
        Provide:
        1. Key patterns identified across sources
        2. Novel combinations of solutions
        3. Academic validation where available
        4. Recent developments and trends
        5. Implementation considerations
        6. Expected impact and risks
        
        Structure your response as a comprehensive synthesis with clear sections.
        """,
        expected_output="Structured synthesis text with multiple sections for each area of analysis",
        agent=agent
    )

def create_report_task(agent, challenge_description, precedents_data, synthesis_text):
    """Create a task for generating a comprehensive report"""
    return Task(
        description=f"""
        Create a comprehensive research report for this challenge:
        
        CHALLENGE:
        {challenge_description}
        
        Use the precedents and synthesis data provided. Format your report with markdown using these sections:
        
        # EXECUTIVE SUMMARY
        [Provide a concise overview of key findings and recommendations]
        
        # 1. CHALLENGE ANALYSIS
        [Break down the key components, constraints, and opportunities]
        
        # 2. PRECEDENT ANALYSIS
        [Analyze the most relevant precedents from different domains]
        
        # 3. INNOVATIVE CONNECTIONS
        [Draw connections between precedents from different domains]
        
        # 4. CROSS-INDUSTRY APPLICATIONS
        [Explore how solutions from other industries can be applied]
        
        # 5. ACTIONABLE RECOMMENDATIONS
        [Provide specific, practical recommendations]
        
        # 6. IMPLEMENTATION ROADMAP
        [Suggest a phased approach for implementing recommendations]
        
        # 7. EXPECTED IMPACT AND RISKS
        [Discuss potential benefits and challenges]
        
        Include references to specific precedents to support your findings.
        """,
        expected_output="Comprehensive markdown formatted research report with all specified sections",
        agent=agent,
        # Fix the context to include the required fields for each item
        context=[
            {
                "role": "precedents", 
                "content": json.dumps(precedents_data, indent=2)[:3000],
                "description": "Precedent data for analysis",
                "expected_output": "Analysis of precedents"
            },
            {
                "role": "synthesis", 
                "content": synthesis_text[:5000],
                "description": "Synthesis of precedents",
                "expected_output": "Integrated insights from synthesis"
            }
        ]
    )

def create_visualization_task(agent, precedents_data):
    """Create a task for visualization code generation"""
    return Task(
        description=f"""
        Create Python code for visualizing the precedents data using matplotlib, networkx, plotly, and streamlit-markmap:
        
        PRECEDENTS DATA:
        {json.dumps(precedents_data, indent=2)[:3000]}
        
        Generate complete, well-commented code for:
        
        1. A network visualization showing relationships between precedents
        2. A pie chart showing distribution of precedents by source type
        3. A timeline visualization of precedents (if date information is available)
        4. A mind map representation of precedents grouped by type and domain
        
        For the mind map, create a markdown-based hierarchical representation that can be rendered with streamlit-markmap.
        Structure it as follows:
        
        ```
        # Innovation Challenge
        ## Academic Precedents
        ### [Domain1]
        - [Precedent Title 1]
        - [Precedent Title 2]
        ### [Domain2]
        - [Precedent Title 3]
        ## News Precedents
        ### [Domain3]
        - [Precedent Title 4]
        ```
        
        The code should be compatible with Streamlit and return visualization objects that can be displayed.
        If a visualization library is not available, include instructions for installation.
        """,
        expected_output="Python code for four different visualizations that can be executed in Streamlit",
        agent=agent
    )

# Streamlit Document Processing Functions
class DocumentProcessor:
    @staticmethod
    def process_pdf(file):
        """Extract text from PDF file"""
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text

    @staticmethod
    def process_docx(file):
        """Extract text from DOCX file"""
        doc = Document(BytesIO(file.getvalue()))
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text

    @staticmethod
    def process_txt(file):
        """Process text file"""
        return file.getvalue().decode("utf-8")

    @staticmethod
    def process_uploaded_files(files):
        """Process multiple uploaded files"""
        all_text = ""
        for file in files:
            if file.type == "application/pdf":
                text = DocumentProcessor.process_pdf(file)
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = DocumentProcessor.process_docx(file)
            elif file.type == "text/plain":
                text = DocumentProcessor.process_txt(file)
            else:
                st.warning(f"Unsupported file type: {file.type}")
                continue
            all_text += text + "\n\n"
        return all_text

# Visualization Functions
class Visualizer:
    @staticmethod
    def create_precedent_network(precedents):
        """Create network visualization of precedent relationships"""
        G = nx.Graph()
        
        # Create nodes with different colors based on source type
        colors = {
            "academic": "lightblue",
            "news": "lightgreen",
            "web": "lightgray",
            "industry": "lightyellow"
        }
        
        node_colors = []
        for p in precedents:
            G.add_node(p.get('title', 'Untitled'), source_type=p.get('type', 'web'))
            node_colors.append(colors.get(p.get('type', 'web'), 'lightgray'))
            
        # Add edges between related precedents
        for i, p1 in enumerate(precedents):
            for j, p2 in enumerate(precedents[i+1:], i+1):
                # Connect nodes if they share domain or type
                if (p1.get('domain') and p1.get('domain') == p2.get('domain')) or \
                   (p1.get('type') and p1.get('type') == p2.get('type')):
                    G.add_edge(p1.get('title', ''), p2.get('title', ''))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)  # For reproducibility
        nx.draw(G, pos, with_labels=True, node_color=node_colors, 
                node_size=1500, font_size=8, font_weight='bold', ax=ax)
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=color, label=source_type, markersize=10)
                          for source_type, color in colors.items()]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        return fig

    @staticmethod
    def create_source_distribution(precedents):
        """Create pie chart of precedent sources"""
        source_counts = {}
        for p in precedents:
            source_type = p.get('type', 'web')
            source_counts[source_type] = source_counts.get(source_type, 0) + 1
            
        fig = px.pie(values=list(source_counts.values()),
                    names=list(source_counts.keys()),
                    title='Distribution of Precedents by Source Type')
        return fig

    @staticmethod
    def create_timeline(precedents):
        """Create timeline of precedents with improved date handling"""
        # First, create a copy of precedents to avoid modifying the original
        processed_precedents = []
        
        # Process and standardize dates
        for p in precedents:
            if not p.get('date') and not p.get('published') and not p.get('year'):
                continue  # Skip precedents without any date information
                
            precedent_copy = p.copy()
            
            # Try to extract date from various possible fields
            date_value = None
            if p.get('date'):
                date_value = p.get('date')
            elif p.get('published'):
                date_value = p.get('published')
            elif p.get('year'):
                # If only year is available, set to January 1st of that year
                date_value = f"{p.get('year')}-01-01"
                
            if date_value:
                precedent_copy['standardized_date'] = date_value
                processed_precedents.append(precedent_copy)
        
        if not processed_precedents:
            return None
            
        # Create DataFrame from processed precedents
        df = pd.DataFrame(processed_precedents)
        
        try:
            # Convert to datetime with flexible parsing
            df['standardized_date'] = pd.to_datetime(df['standardized_date'], errors='coerce', infer_datetime_format=True)
            df = df.dropna(subset=['standardized_date'])  # Remove rows with invalid dates
            
            if df.empty:
                return None
                
            # For visualization, we'll use a scatter plot which doesn't require end dates
            fig = px.scatter(df, x='standardized_date', y='title', color='type',
                            size=[10] * len(df),  # Consistent point size
                            title='Precedents Timeline',
                            labels={'standardized_date': 'Date', 'title': 'Precedent'})
            
            # Improve layout
            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Precedent',
                height=max(400, 50 + (len(df) * 30)),
                margin=dict(l=20, r=20, t=40, b=20),
                legend_title="Source Type"
            )
            
            # Customize points
            fig.update_traces(
                marker=dict(size=15, symbol='circle', line=dict(width=1, color='black')),
                mode='markers+text',
                textposition='middle right'
            )
            
            return fig
        except Exception as e:
            print(f"Detailed timeline error: {str(e)}")
            
            # Try an even simpler approach if the above fails
            try:
                # Just create a very basic table as fallback
                fig = go.Figure(data=[go.Table(
                    header=dict(values=['Title', 'Date', 'Source Type'],
                                fill_color='paleturquoise',
                                align='left'),
                    cells=dict(values=[
                        df['title'].tolist(),
                        df['standardized_date'].dt.strftime('%Y-%m-%d').tolist(),
                        df['type'].tolist()
                    ],
                    fill_color='lavender',
                    align='left')
                )])
                return fig
            except Exception as e2:
                print(f"Even the fallback timeline failed: {str(e2)}")
                return None

    @staticmethod
    def create_mind_map_content(precedents, challenge_title="Innovation Challenge"):
        """Create markdown content for the mind map"""
        mind_map_content = f"# {challenge_title}\n"
        
        # Group precedents by type
        precedents_by_type = {}
        for p in precedents:
            p_type = p.get('type', 'general')
            if p_type not in precedents_by_type:
                precedents_by_type[p_type] = []
            precedents_by_type[p_type].append(p)
        
        # Add each type and its precedents to the mind map
        for p_type, type_precedents in precedents_by_type.items():
            mind_map_content += f"## {p_type.title()} Precedents\n"
            
            # Group by domain if available
            domains = {}
            for p in type_precedents:
                domain = p.get('domain', 'General')
                if domain not in domains:
                    domains[domain] = []
                domains[domain].append(p)
            
            # Add each domain and its precedents
            for domain, domain_precedents in domains.items():
                mind_map_content += f"### {domain}\n"
                for p in domain_precedents:
                    title = p.get('title', 'Untitled Precedent')
                    # Truncate title if it's too long
                    if len(title) > 40:
                        title = title[:37] + "..."
                    
                    mind_map_content += f"- {title}\n"
        
        return mind_map_content

# Export Functions
class Exporter:
    @staticmethod
    def to_excel(data):
        """Export data to Excel (CSV)"""
        df = pd.DataFrame(data)
        return df.to_csv(index=False).encode('utf-8')

    @staticmethod
    def to_json(data):
        """Export data to JSON"""
        return json.dumps(data, indent=4)

# API Configuration
def api_config_section():
    """Configure API keys in sidebar"""
    with st.sidebar.expander("API Configuration", expanded=True):
        # Google API Key (for Gemini)
        google_key = st.text_input("Google API Key", 
            type="password",
            value=st.session_state.get('google_key', ''),
            help="Required for Gemini model")
        if google_key:
            st.session_state.google_key = google_key
            os.environ["GOOGLE_API_KEY"] = google_key
            st.sidebar.write(f"Google API key set: {google_key[:5]}..." if len(google_key) > 5 else "Invalid key")

        # News API Key
        news_key = st.text_input("News API Key", 
            type="password",
            value=st.session_state.get('news_api_key', ''),
            help="For news search")
        if news_key:
            st.session_state.news_api_key = news_key

        # Serper API Key
        serper_key = st.text_input("Serper API Key", 
            type="password",
            value=st.session_state.get('serper_key', ''),
            help="For enhanced web search")
        if serper_key:
            st.session_state.serper_key = serper_key

        if st.button("Save API Keys"):
            st.success("API keys saved successfully!")

# CrewAI Execution Functions
def run_precedent_search(challenge_description):
    """Run the precedent search crew"""
    explorer_agent = create_precedents_explorer_agent()
    search_task = create_precedent_search_task(explorer_agent, challenge_description)
    
    search_crew = Crew(
        agents=[explorer_agent],
        tasks=[search_task],
        process=Process.sequential,
        verbose=True
    )
    
    with st.spinner("Searching for precedents across multiple sources..."):
        # Get the result from the crew kickoff
        result = search_crew.kickoff()
        
        # Extract the string content from the CrewOutput object
        if hasattr(result, 'raw_output'):
            result_text = result.raw_output
        else:
            # Fall back to string representation if raw_output isn't available
            result_text = str(result)
        
        # Parse the result as JSON if possible
        try:
            precedents = json.loads(result_text)
            return precedents
        except json.JSONDecodeError:
            # If we can't parse as JSON, try to extract structured data from text
            return parse_text_result(result_text)

def run_synthesis(challenge_description, precedents):
    """Run the synthesis crew"""
    synthesizer_agent = create_synthesizer_agent()
    synthesis_task = create_synthesis_task(synthesizer_agent, challenge_description, precedents)
    
    synthesis_crew = Crew(
        agents=[synthesizer_agent],
        tasks=[synthesis_task],
        process=Process.sequential,
        verbose=True
    )
    
    with st.spinner("Synthesizing innovative solutions from precedents..."):
        result = synthesis_crew.kickoff()
        # Extract the string content from the CrewOutput object
        if hasattr(result, 'raw_output'):
            return result.raw_output
        else:
            return str(result)

def run_report_generation(challenge_description, precedents, synthesis_text):
    """Run the report generation crew"""
    report_agent = create_report_agent()
    report_task = create_report_task(report_agent, challenge_description, precedents, synthesis_text)
    
    report_crew = Crew(
        agents=[report_agent],
        tasks=[report_task],
        process=Process.sequential,
        verbose=True
    )
    
    with st.spinner("Generating comprehensive research report..."):
        result = report_crew.kickoff()
        # Extract the string content from the CrewOutput object
        if hasattr(result, 'raw_output'):
            return result.raw_output
        else:
            return str(result)

def run_visualization(precedents):
    """Run the visualization crew"""
    visualization_agent = create_visualization_agent()
    visualization_task = create_visualization_task(visualization_agent, precedents)
    
    visualization_crew = Crew(
        agents=[visualization_agent],
        tasks=[visualization_task],
        process=Process.sequential,
        verbose=True
    )
    
    with st.spinner("Creating visualizations for precedent analysis..."):
        result = visualization_crew.kickoff()
        # Extract the string content from the CrewOutput object
        if hasattr(result, 'raw_output'):
            return result.raw_output
        else:
            return str(result)

# Helper Functions
def parse_text_result(text_result):
    """Parse text result into structured precedent format with improved robustness"""
    precedents = []
    try:
        # Add check for None
        if text_result is None:
            st.error("Received empty response from agent")
            return [{
                "title": "Error in Analysis",
                "description": "No response was received from the agent.",
                "domain": "General",
                "type": "general",
                "relevance": "medium",
                "source_url": "",
                "date": ""
            }]
        
        # Try to parse as JSON first
        try:
            # Look for JSON-like content within the text
            import re
            json_pattern = r'\[\s*{.*}\s*\]'  # Pattern to find JSON arrays
            json_matches = re.findall(json_pattern, text_result, re.DOTALL)
            
            if json_matches:
                for match in json_matches:
                    try:
                        json_data = json.loads(match)
                        if isinstance(json_data, list):
                            return json_data
                    except:
                        continue
            
            # Try direct parsing if pattern matching failed
            json_data = json.loads(text_result)
            if isinstance(json_data, list):
                return json_data
            elif isinstance(json_data, dict):
                return [json_data]
        except json.JSONDecodeError:
            # Not valid JSON, continue with text parsing
            pass
            
        # Enhanced text parsing logic
        current_precedent = {}
        lines = text_result.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # New precedent starts with number, "Precedent" keyword, or title pattern
            if (line.startswith(('1.', '2.', '3.', '4.', '5.', 'Precedent')) or 
                'Title:' in line or 'TITLE:' in line):
                if current_precedent and 'title' in current_precedent:
                    precedents.append(current_precedent)
                current_precedent = {}
                
                # Extract title if it's on this line
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    if 'title' in key:
                        current_precedent['title'] = value.strip()
                    else:
                        current_precedent['title'] = line
                else:
                    current_precedent['title'] = line
                    
            # Extract key information
            elif ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if 'title' in key:
                    current_precedent['title'] = value
                elif 'desc' in key or 'description' in key:
                    current_precedent['description'] = value
                elif 'domain' in key:
                    current_precedent['domain'] = value
                elif 'type' in key:
                    current_precedent['type'] = value.lower()  # Normalize type
                elif 'relev' in key:
                    current_precedent['relevance'] = value
                elif 'source' in key or 'url' in key:
                    current_precedent['source_url'] = value
                    
        # Add the last precedent if not empty
        if current_precedent and 'title' in current_precedent:
            precedents.append(current_precedent)
            
        # If no precedents were parsed, create one from the entire text
        if not precedents:
            precedents.append({
                "title": "AI-Generated Analysis",
                "description": text_result[:500],
                "domain": "General",
                "type": "general",
                "relevance": "medium",
                "source_url": "",
                "date": ""
            })
            
        # Ensure required fields
        for p in precedents:
            if 'description' not in p:
                p['description'] = "No description available"
            if 'type' not in p:
                p['type'] = "general"
    except Exception as e:
        st.error(f"Error parsing text result: {str(e)}")
        # Create a single precedent from the whole text as fallback
        precedents.append({
            "title": "AI-Generated Analysis",
            "description": text_result[:500] if isinstance(text_result, str) else str(text_result)[:500],
            "domain": "General",
            "type": "general",
            "relevance": "medium",
            "source_url": "",
            "date": ""
        })
    
    return precedents

# Main Application
def main():
    st.set_page_config(page_title="Precedents Thinking System", layout="wide")
    st.title("Research-Intensive Precedents Thinking System")
    st.markdown("*Find innovative cross-industry solutions to complex challenges through precedent analysis*")

    # API Configuration in sidebar
    api_config_section()

    # File upload in sidebar
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload relevant documents", 
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt']
        )
        
        if uploaded_files:
            document_text = DocumentProcessor.process_uploaded_files(uploaded_files)
            st.session_state.document_context = document_text
            st.success(f"Processed {len(uploaded_files)} documents")

    # Main navigation
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Challenge Framing", 
        "Precedent Search", 
        "Idea Synthesis", 
        "Visualization", 
        "Export", 
        "Research Report"
    ])

    # Challenge Framing Tab
    with tab1:
        st.header("Challenge Framing")
        
        if 'document_context' in st.session_state:
            with st.expander("📄 Uploaded Document Context"):
                st.text_area("Document Content", st.session_state.document_context, height=200)
        
        with st.form("challenge_form"):
            challenge = st.text_area(
                "Describe your challenge:",
                help="Be specific about what you're trying to achieve",
                height=150
            )
            
            domain = st.text_input(
                "Primary domain/industry:",
                help="What industry or field is this challenge in?"
            )
            
            search_options = st.multiselect(
                "Search Sources",
                ["Academic Papers", "News Articles", "Web Search", "Industry Cases"],
                default=["Academic Papers", "News Articles", "Web Search", "Industry Cases"],
                help="Select which sources to include in the search"
            )
            
            use_docs = st.checkbox("Include uploaded documents in analysis", value=True)
            if st.form_submit_button("Frame Challenge"):
                full_context = challenge
                if use_docs and 'document_context' in st.session_state:
                    full_context += "\n\nAdditional Context from Documents:\n" + st.session_state.document_context
                
                st.session_state.framed_challenge = {
                    "description": full_context,
                    "domain": domain,
                    "search_options": search_options
                }
                st.success("Challenge framed successfully!")

    # Precedent Search Tab
    with tab2:
        st.header("Precedent Search")
        
        if not st.session_state.get('framed_challenge'):
            st.warning("Please frame your challenge first")
        else:
            if st.button("Search Precedents"):
                try:
                    # Run the precedent search crew
                    precedents = run_precedent_search(st.session_state.framed_challenge["description"])
                    
                    # Store results in session state
                    st.session_state.precedents = precedents
                    
                    # Debug info
                    st.write(f"Total precedents found: {len(precedents)}")
                    if precedents:
                        # Display types distribution
                        types = {}
                        for p in precedents:
                            t = p.get('type', 'None')
                            types[t] = types.get(t, 0) + 1
                        st.write("Distribution by type:", types)
                    
                    # Ensure all precedents have a type
                    for p in precedents:
                        if 'type' not in p or not p['type']:
                            p['type'] = 'web'  # Default type
                    
                    # Display by source type
                    for source_type in ['academic', 'news', 'web', 'industry']:
                        source_precedents = [p for p in precedents if p.get('type') == source_type]
                        if source_precedents:
                            with st.expander(f"📚 {source_type.title()} Sources ({len(source_precedents)})", expanded=source_type == 'academic'):
                                for p in source_precedents:
                                    st.markdown(f"### {p['title']}")
                                    st.markdown(f"**Description:** {p['description']}")
                                    st.markdown(f"**Relevance:** {p.get('relevance', 'Medium')}")
                                    
                                    # URL handling
                                    if 'source_url' in p and p['source_url']:
                                        st.markdown(f"**Source:** [{p['source_url']}]({p['source_url']})")
                                    elif 'url' in p and p['url']:
                                        st.markdown(f"**Source:** [{p['url']}]({p['url']})")
                                    else:
                                        st.markdown("**Source:** Not available")
                                        
                                    if p.get('date'):
                                        st.markdown(f"**Date:** {p['date']}")
                                    st.markdown("---")
                    
                    # If no precedents were found
                    if not precedents:
                        st.warning("No relevant precedents found. Consider refining your challenge description or search options.")
                except Exception as e:
                    st.error(f"Error during precedent search: {str(e)}")
                    st.info("Try again or refine your challenge description.")

    # Idea Synthesis Tab
    with tab3:
        st.header("Idea Synthesis")
        
        if not st.session_state.get('precedents'):
            st.warning("Please search for precedents first")
        else:
            if st.button("Synthesize Ideas"):
                try:
                    # Run the synthesis crew
                    synthesis_text = run_synthesis(
                        st.session_state.framed_challenge["description"],
                        st.session_state.precedents
                    )
                    
                    # Store the raw synthesis text
                    st.session_state.synthesis_text = synthesis_text
                    
                    # Display the synthesis
                    st.subheader("💡 Complete Synthesis")
                    st.markdown(synthesis_text)
                    
                    # Try to extract sections using flexible approach
                    st.subheader("📋 Synthesis by Sections")
                    
                    # Section markers to identify in the text
                    section_markers = [
                        ("🔍 Key Patterns", ["key pattern", "pattern", "1."]),
                        ("💡 Novel Combinations", ["novel combination", "combination", "2."]),
                        ("📚 Academic Validation", ["academic validation", "validation", "3."]),
                        ("📈 Recent Developments", ["recent development", "trend", "4."]),
                        ("🎯 Implementation Considerations", ["implementation", "5."]),
                        ("📊 Expected Impact", ["impact", "risk", "6."])
                    ]
                    
                    # Split by lines and look for section headers
                    lines = synthesis_text.split('\n')
                    current_section = None
                    section_content = {}
                    
                    for line in lines:
                        # Check if this line is a section header
                        for title, markers in section_markers:
                            if any(marker.lower() in line.lower() for marker in markers):
                                current_section = title
                                if current_section not in section_content:
                                    section_content[current_section] = []
                                break
                        
                        # Add line to current section if we're in one
                        if current_section and line.strip():
                            section_content[current_section].append(line)
                    
                    # Display each section
                    for title, markers in section_markers:
                        if title in section_content:
                            with st.expander(title, expanded=True):
                                st.markdown('\n'.join(section_content[title]))
                        else:
                            # Try to find content matching these keywords
                            related_content = []
                            for line in lines:
                                if any(marker.lower() in line.lower() for marker in markers):
                                    related_content.append(line)
                            
                            if related_content:
                                with st.expander(title):
                                    st.markdown('\n'.join(related_content))
                except Exception as e:
                    st.error(f"Error during synthesis: {str(e)}")
                    st.info("Try again or check if you have enough diverse precedents.")

    # Visualization Tab
    with tab4:
        st.header("Precedent Visualizations")
        
        if not st.session_state.get('precedents'):
            st.warning("Please search for precedents first")
        else:
            # Set a wider layout specifically for the visualization
            st.markdown("""
            <style>
            .mindmap-container {
                width: 100%;
                overflow-x: auto;
                min-height: 10px;
            }
            .viz-container {
                width: 100%;
                margin-top: 20px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            if st.button("Generate Visualizations") or 'visualization_code' in st.session_state:
                try:
                    # Run the visualization agent if we haven't already
                    if 'visualization_code' not in st.session_state:
                        visualization_code = run_visualization(st.session_state.precedents)
                        st.session_state.visualization_code = visualization_code
                    
                    # Display tabs for different visualizations
                    viz_tabs = st.tabs(["Mind Map", "Network Graph", "Source Distribution", "Timeline"])
                    
                    # Mind Map Visualization
                    with viz_tabs[0]:
                        st.subheader("🧠 Interactive Mind Map of Precedents")
                        
                        # Import the required library
                        try:
                            from streamlit_markmap import markmap
                        except ImportError:
                            st.error("The streamlit-markmap package is not installed. Please install it with: pip install streamlit-markmap==1.0.1")
                            st.code("pip install streamlit-markmap==1.0.1", language="bash")
                            
                            # Provide the markdown content as fallback
                            mind_map_content = Visualizer.create_mind_map_content(st.session_state.precedents)
                            st.code(mind_map_content, language="markdown")
                        else:
                            # Use the mind map content from visualization agent or generate it
                            try:
                                # Search for mind map in the visualization code
                                mind_map_content = None
                                lines = st.session_state.visualization_code.split('\n')
                                in_mind_map_section = False
                                mind_map_lines = []
                                
                                for line in lines:
                                    # Look for sections that define mind map content
                                    if "mind_map_content" in line and "=" in line:
                                        in_mind_map_section = True
                                        if "=" in line:
                                            mind_map_lines.append(line.split("=", 1)[1].strip())
                                    elif in_mind_map_section and (line.strip().startswith('"') or line.strip().startswith("'") or 
                                                                line.strip().startswith("f'") or line.strip().startswith('f"')):
                                        mind_map_lines.append(line.strip())
                                    elif in_mind_map_section and (line.strip().startswith("+= ") or "+=" in line):
                                        if "+=" in line:
                                            mind_map_lines.append(line.split("+=", 1)[1].strip())
                                    elif in_mind_map_section and not line.strip():
                                        continue
                                    elif in_mind_map_section and not (line.strip().startswith("#") or 
                                                                    "mind_map_content" in line):
                                        in_mind_map_section = False
                                
                                if mind_map_lines:
                                    # Combine all lines and clean up
                                    mind_map_text = " ".join(mind_map_lines)
                                    # Remove quotes and escape characters
                                    mind_map_text = mind_map_text.replace('"""', '').replace("'''", '')
                                    mind_map_text = mind_map_text.strip('"').strip("'")
                                    mind_map_text = mind_map_text.replace("\\n", "\n")
                                    mind_map_content = mind_map_text
                                
                                # If we couldn't extract it, generate it
                                if not mind_map_content:
                                    st.info("Generating mind map from precedent data...")
                                    mind_map_content = Visualizer.create_mind_map_content(st.session_state.precedents)
                                    
                                # Display the mind map with the generated markdown
                                st.markdown('<div class="mindmap-container">', unsafe_allow_html=True)
                                markmap(mind_map_content)
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Add instructions
                                st.markdown("""
                                ### Using the Mind Map
                                
                                This interactive mind map shows relationships between different precedent categories:
                                
                                - **Click on nodes** to expand or collapse sections
                                - **Drag nodes** to rearrange the visualization
                                - **Zoom in/out** using the mouse wheel
                                - **Pan** by clicking and dragging the background
                                
                                If the mind map appears cut off, you can:
                                1. Click on nodes to collapse sections you're not currently viewing
                                2. Drag the map to center the section you want to view
                                3. Use your browser's zoom out feature (Ctrl/Cmd + -) to see more of the map at once
                                """)
                            except Exception as e:
                                st.error(f"Error generating mind map: {str(e)}")
                                # Fall back to basic visualization
                                st.info("Generating basic mind map...")
                                mind_map_content = Visualizer.create_mind_map_content(st.session_state.precedents)
                                markmap(mind_map_content)
                    
                    # Network Graph Visualization
                    with viz_tabs[1]:
                        st.subheader("🕸️ Network Graph of Precedent Relationships")
                        
                        try:
                            # Check if we have code from the visualization agent
                            if 'network_graph' not in st.session_state:
                                # Use our built-in function as fallback
                                network_fig = Visualizer.create_precedent_network(st.session_state.precedents)
                                st.session_state.network_graph = network_fig
                            
                            # Display the network graph
                            st.pyplot(st.session_state.network_graph)
                            
                            # Add explanation
                            st.markdown("""
                            ### Understanding the Network Graph
                            
                            This network visualization shows connections between precedents:
                            
                            - **Nodes** represent individual precedents
                            - **Colors** indicate different source types:
                            - **Light Blue**: Academic sources
                            - **Light Green**: News sources
                            - **Light Gray**: Web sources
                            - **Light Yellow**: Industry cases
                            - **Connections** show relationships between precedents that share domains or types
                            
                            Closely connected precedents may offer complementary insights or represent a cohesive trend.
                            """)
                        except Exception as e:
                            st.error(f"Error generating network graph: {str(e)}")
                            st.info("Try again with more diverse precedents to see relationships.")
                    
                    # Source Distribution Visualization
                    with viz_tabs[2]:
                        st.subheader("📊 Distribution of Precedents by Source Type")
                        
                        try:
                            # Check if we have code from the visualization agent
                            if 'source_distribution' not in st.session_state:
                                # Use our built-in function as fallback
                                dist_fig = Visualizer.create_source_distribution(st.session_state.precedents)
                                st.session_state.source_distribution = dist_fig
                            
                            # Display the pie chart
                            st.plotly_chart(st.session_state.source_distribution, use_container_width=True)
                            
                            # Get counts for text summary
                            source_counts = {}
                            for p in st.session_state.precedents:
                                source_type = p.get('type', 'web')
                                source_counts[source_type] = source_counts.get(source_type, 0) + 1
                            
                            # Add explanation
                            st.markdown("### Source Distribution Analysis")
                            
                            # Show text summary
                            st.markdown("**Summary of Source Types:**")
                            for source_type, count in source_counts.items():
                                percentage = (count / len(st.session_state.precedents)) * 100
                                st.markdown(f"- **{source_type.title()}**: {count} precedents ({percentage:.1f}%)")
                            
                            # Interpretation
                            highest_source = max(source_counts.items(), key=lambda x: x[1])[0]
                            st.markdown(f"""
                            **Interpretation:** The majority of precedents come from **{highest_source}** sources. 
                            This distribution affects the balance of theoretical versus practical insights in your analysis.
                            """)
                        except Exception as e:
                            st.error(f"Error generating source distribution: {str(e)}")
                            st.info("Make sure your precedents have 'type' attributes defined.")
                    
                    # Timeline Visualization
                    with viz_tabs[3]:
                        st.subheader("⏱️ Timeline of Precedents")
                        
                        try:
                            # Check if precedents have any kind of date information
                            date_fields = ['date', 'published', 'year']
                            dated_precedents = [p for p in st.session_state.precedents 
                                            if any(p.get(field) for field in date_fields)]
                            
                            if not dated_precedents:
                                st.info("No date information available in the precedents. Timeline cannot be created.")
                                
                                # Show what date formats are accepted
                                st.markdown("""
                                ### Adding Date Information
                                
                                To create a timeline visualization, your precedents need date information in one of these formats:
                                
                                1. A `date` field: "2023-05-15", "05/15/2023", "May 15, 2023"
                                2. A `published` field (for publications): "2023-05-15T14:30:00Z"
                                3. A `year` field (for historical precedents): "2023"
                                
                                You can add this information by:
                                - Exporting your precedents as JSON
                                - Adding date fields in your preferred format
                                - Reuploading the file
                                """)
                                
                                # Show a sample of the precedents to help user understand the current structure
                                sample_precedent = st.session_state.precedents[0] if st.session_state.precedents else {}
                                st.markdown("#### Sample Precedent Structure")
                                st.json(sample_precedent)
                            else:
                                # Generate timeline
                                timeline_fig = Visualizer.create_timeline(st.session_state.precedents)
                                
                                if timeline_fig:
                                    st.plotly_chart(timeline_fig, use_container_width=True)
                                    
                                    # Add explanation
                                    st.markdown("""
                                    ### Timeline Analysis
                                    
                                    This timeline shows the chronological distribution of precedents:
                                    
                                    - **Each point** represents a precedent
                                    - **Colors** indicate different source types
                                    - **Position** shows when each precedent was published
                                    
                                    The timeline helps identify how solutions and approaches have evolved over time.
                                    Recent precedents may represent current best practices, while older ones may show foundational concepts.
                                    """)
                                else:
                                    st.warning("Timeline creation failed. Check date formats in your precedents.")
                                    
                                    # Diagnostic information to help troubleshoot
                                    st.markdown("#### Date Format Diagnostic")
                                    st.markdown("Here are the date values found in your precedents:")
                                    
                                    date_info = []
                                    for i, p in enumerate(st.session_state.precedents[:10]):  # Show first 10 only
                                        date_info.append({
                                            'index': i,
                                            'title': p.get('title', 'Untitled'),
                                            'date_field': p.get('date', None),
                                            'published_field': p.get('published', None),
                                            'year_field': p.get('year', None)
                                        })
                                    
                                    if date_info:
                                        st.table(date_info)
                                        st.markdown("""
                                        **Common Date Format Issues:**
                                        - Missing date information
                                        - Inconsistent formats (mixing "YYYY-MM-DD" with "Month Day, Year")
                                        - Invalid dates (e.g., "February 30")
                                        - Extra characters in date strings
                                        
                                        Try standardizing your date formats and ensure they're valid dates.
                                        """)
                                    else:
                                        st.markdown("No date information found in the precedents.")
                        except Exception as e:
                            st.error(f"Error analyzing timeline data: {str(e)}")
                            st.info("The visualization system encountered an unexpected error. Please try with different data.")
                    
                    # Add word cloud as a bonus visualization if we have enough precedents
                    if len(st.session_state.precedents) >= 5:
                        st.subheader("🔤 Key Terms Word Cloud")
                        
                        try:
                            # Extract text from precedent titles and descriptions
                            all_text = " ".join([
                                p.get('title', '') + " " + p.get('description', '')
                                for p in st.session_state.precedents
                            ])
                            
                            # Generate word cloud
                            wordcloud = WordCloud(
                                width=800, 
                                height=400, 
                                background_color='white',
                                colormap='viridis',
                                max_words=100,
                                contour_width=3,
                                contour_color='steelblue'
                            ).generate(all_text)
                            
                            # Display the word cloud
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            st.pyplot(fig)
                            
                            # Add explanation
                            st.markdown("""
                            ### Word Cloud Analysis
                            
                            This word cloud highlights key terms from all precedent titles and descriptions. 
                            Larger words appear more frequently in the data and may represent central themes
                            across different sources and domains.
                            """)
                        except Exception as e:
                            st.error(f"Error generating word cloud: {str(e)}")
                
                except Exception as e:
                    st.error(f"Error generating visualizations: {str(e)}")
                    st.info("Try running with different precedents or check precedent data structure.")
            else:
                st.info("Click 'Generate Visualizations' to create interactive visualizations from your precedents.")

    # Export Tab
    with tab5:
        st.header("Export Results")
        
        if not st.session_state.get('precedents'):
            st.warning("Please search for precedents first")
        else:
            export_format = st.selectbox(
                "Export Format",
                ["Excel (CSV)", "JSON", "Research Report"]
            )

            if st.button("Export"):
                if export_format == "Excel (CSV)":
                    data = Exporter.to_excel(st.session_state.precedents)
                    st.download_button(
                        label="Download Excel (CSV)",
                        data=data,
                        file_name="precedents.csv",
                        mime="text/csv"
                    )
                elif export_format == "JSON":
                    data = Exporter.to_json(st.session_state.precedents)
                    st.download_button(
                        label="Download JSON",
                        data=data,
                        file_name="precedents.json",
                        mime="application/json"
                    )
                else:
                    # Get synthesis if available, otherwise run synthesis
                    if not st.session_state.get('synthesis_text'):
                        st.info("No synthesis available. Generating synthesis first...")
                        synthesis_text = run_synthesis(
                            st.session_state.framed_challenge["description"],
                            st.session_state.precedents
                        )
                        st.session_state.synthesis_text = synthesis_text
                    
                    # Generate and download report
                    report = run_report_generation(
                        st.session_state.framed_challenge["description"],
                        st.session_state.precedents,
                        st.session_state.synthesis_text
                    )
                    
                    st.download_button(
                        label="Download Research Report",
                        data=report,
                        file_name="precedents_research_report.md",
                        mime="text/markdown"
                    )

    # Research Report Tab
    with tab6:
        st.header("Research Report")
        
        if not st.session_state.get('precedents'):
            st.warning("Please search for precedents first")
        else:
            report_options = st.multiselect(
                "Report Options",
                ["Include Network Visualization", "Include Source Analysis", "Include Cross-Industry Applications"],
                default=["Include Network Visualization", "Include Cross-Industry Applications"]
            )
            
            if st.button("Generate Report"):
                try:
                    # Get synthesis if available, otherwise run synthesis
                    if not st.session_state.get('synthesis_text'):
                        st.info("No synthesis available. Generating synthesis first...")
                        synthesis_text = run_synthesis(
                            st.session_state.framed_challenge["description"],
                            st.session_state.precedents
                        )
                        st.session_state.synthesis_text = synthesis_text
                    
                    # Generate comprehensive report
                    report = run_report_generation(
                        st.session_state.framed_challenge["description"],
                        st.session_state.precedents,
                        st.session_state.synthesis_text
                    )
                    
                    # Store the report in session state
                    st.session_state.report = report
                    
                    # Display sections of the report using st.markdown
                    sections = report.split('#')
                    
                    # Process each section
                    for i, section in enumerate(sections):
                        if i == 0:  # Skip the first empty split
                            continue
                            
                        # Add back the # that was removed in the split
                        formatted_section = '#' + section
                        
                        # Display each section with proper markdown rendering
                        st.markdown(formatted_section)
                    
                    # Provide download option
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name="precedents_research_report.md",
                        mime="text/markdown"
                    )
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
                    st.info("Try again or check that you have sufficient precedent data.")

    # Feedback section in sidebar
    with st.sidebar.expander("📝 Provide Feedback"):
        rating = st.slider("Rate your experience", 1, 5, 3)
        feedback = st.text_area("Additional comments")
        if st.button("Submit Feedback"):
            st.session_state.feedback_data.append({
                "rating": rating,
                "feedback": feedback,
                "timestamp": datetime.now().isoformat()
            })
            st.sidebar.success("Thank you for your feedback!")

if __name__ == "__main__":
    main()            
