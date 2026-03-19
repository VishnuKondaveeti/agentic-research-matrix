"""
LLM-powered generation using LangChain and Google Gemini.
Provides prompt templates for Q&A, summarization, and report generation.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.settings import settings


class Generator:
    """LLM-powered response generation for the RAG system."""

    def __init__(self, llm_provider: str = None):
        self.llm = self._init_llm(llm_provider)
        self.parser = StrOutputParser()

    def _init_llm(self, provider_override: str = None):
        """Initialize the language model based on configured provider."""
        provider = (provider_override or settings.llm_provider).lower()

        if provider == "gemini":
            if not settings.google_api_key:
                raise ValueError("GOOGLE_API_KEY is required for Gemini provider.")
            return ChatGoogleGenerativeAI(
                model=settings.llm_model,
                google_api_key=settings.google_api_key,
                temperature=0.3,
                max_output_tokens=4096,
            )
        
        elif provider == "ollama":
            return ChatOllama(
                model=settings.ollama_model,
                base_url=settings.ollama_host,
                temperature=0.3,
            )
            
        elif provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI provider.")
            return ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                temperature=0.3,
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    # ── Prompt Templates ──

    QA_TEMPLATE = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a research assistant specializing in academic literature analysis. "
            "Answer the user's question based ONLY on the provided research context. "
            "If the context doesn't contain enough information, say so clearly. "
            "Cite specific papers when possible. Be precise and academic in tone. "
            "Tone/Complexity: {complexity_instruction}"
        )),
        ("human", (
            "Research Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Provide a detailed, well-structured answer based on the research context above."
        )),
    ])

    SUMMARY_TEMPLATE = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert research summarizer. Create concise, "
            "informative summaries that capture key findings, methodology, "
            "and contributions of research papers. "
            "Tone/Complexity: {complexity_instruction}"
        )),
        ("human", (
            "Research Content:\n{context}\n\n"
            "Create a comprehensive summary covering:\n"
            "1. Main research question/objective\n"
            "2. Methodology used\n"
            "3. Key findings and results\n"
            "4. Contributions and significance\n"
            "5. Limitations mentioned"
        )),
    ])

    REPORT_TEMPLATE = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a senior research analyst. Generate structured literature review "
            "reports suitable for academic publications. Use formal academic language "
            "and proper structure."
        )),
        ("human", (
            "Research Topic: {topic}\n\n"
            "Research Context:\n{context}\n\n"
            "Generate a comprehensive literature review report with these sections:\n"
            "# Literature Review: {topic}\n\n"
            "## 1. Introduction\n"
            "Brief overview of the research area and its importance.\n\n"
            "## 2. Related Work\n"
            "Survey of key research contributions.\n\n"
            "## 3. Key Research Approaches\n"
            "Major methodologies and techniques identified.\n\n"
            "## 4. Current Trends\n"
            "Emerging patterns and active research directions.\n\n"
            "## 5. Research Gaps\n"
            "Identified gaps and unexplored areas.\n\n"
            "## 6. Future Directions\n"
            "Suggested future research opportunities.\n\n"
            "## 7. Conclusion\n"
            "Summary of key takeaways."
        )),
    ])

    ADVISOR_TEMPLATE = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a research advisor with deep expertise across multiple fields. "
            "Analyze the research landscape and identify gaps, opportunities, and "
            "novel research directions."
        )),
        ("human", (
            "Research Topic: {topic}\n\n"
            "Existing Research Context:\n{context}\n\n"
            "Based on the existing research, provide:\n"
            "1. **Research Gaps**: Areas that are under-explored\n"
            "2. **Novel Ideas**: Potential new research directions\n"
            "3. **Cross-Domain Opportunities**: Ideas connecting different fields\n"
            "4. Methodological Suggestions**: New approaches worth trying\n"
            "5. Priority Ranking**: Which ideas have the most potential impact"
        )),
    ])

    CRITIQUE_TEMPLATE = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a critical research reviewer. Evaluate the provided literature review report "
            "for accuracy, hallucinations, proper formatting, and citation coverage based on the context. "
            "Provide a series of critiques and then a corrected version of the report."
        )),
        ("human", (
            "Research Topic: {topic}\n\n"
            "Research Context:\n{context}\n\n"
            "Draft Report:\n{report}\n\n"
            "Critique the report and provide an improved, final version. "
            "Formatting should remain as a structured markdown report."
        )),
    ])

    ANALYSIS_TEMPLATE = ChatPromptTemplate.from_messages([

        ("system", (
            "You are a research paper analyst. Provide detailed analysis "
            "extracting key insights, methodology assessments, and critical evaluations. "
            "Complexity Target: {complexity_instruction}"
        )),
        ("human", (
            "Paper Content:\n{context}\n\n"
            "Provide a detailed analysis covering:\n"
            "1. **Key Findings**: Main results and discoveries\n"
            "2. **Methodology Assessment**: Strengths and weaknesses of the approach\n"
            "3. **Contributions**: Novel contributions to the field\n"
            "4. **Limitations**: Identified limitations and potential issues\n"
            "5. **Practical Implications**: Real-world applications\n"
            "6. **Relation to Prior Work**: How this connects to existing research\n"
            "7. **GitHub/Code Links**: Any extracted repository URLs\n"
            "8. **Datasets**: Any named datasets mentioned"
        )),
    ])

    COMPLEXITY_INSTRUCTIONS = {
        "beginner": "Explain as if to someone new to the field. Avoid overly technical jargon, define complex terms, and use analogies where helpful.",
        "expert": "Provide a high-level technical analysis suitable for a peer-review or senior researcher. Use precise academic terminology and focus on nuanced details.",
        "standard": "Provide a balanced academic explanation suitable for a general research audience."
    }

    # ── Generation Methods ──

    def generate_answer(self, context: str, question: str, complexity: str = "standard") -> str:
        """Generate an answer to a question given research context."""
        instruction = self.COMPLEXITY_INSTRUCTIONS.get(complexity, self.COMPLEXITY_INSTRUCTIONS["standard"])
        chain = self.QA_TEMPLATE | self.llm | self.parser
        return chain.invoke({"context": context, "question": question, "complexity_instruction": instruction})

    def generate_summary(self, context: str, complexity: str = "standard") -> str:
        """Generate a summary of research content."""
        instruction = self.COMPLEXITY_INSTRUCTIONS.get(complexity, self.COMPLEXITY_INSTRUCTIONS["standard"])
        chain = self.SUMMARY_TEMPLATE | self.llm | self.parser
        return chain.invoke({"context": context, "complexity_instruction": instruction})

    def generate_report(self, topic: str, context: str) -> str:
        """Generate a literature review report."""
        chain = self.REPORT_TEMPLATE | self.llm | self.parser
        return chain.invoke({"topic": topic, "context": context})

    def generate_advice(self, topic: str, context: str) -> str:
        """Generate research advice identifying gaps and ideas."""
        chain = self.ADVISOR_TEMPLATE | self.llm | self.parser
        return chain.invoke({"topic": topic, "context": context})

    def generate_analysis(self, context: str, complexity: str = "standard") -> str:
        """Generate a detailed analysis of paper content."""
        instruction = self.COMPLEXITY_INSTRUCTIONS.get(complexity, self.COMPLEXITY_INSTRUCTIONS["standard"])
        chain = self.ANALYSIS_TEMPLATE | self.llm | self.parser
        return chain.invoke({"context": context, "complexity_instruction": instruction})

    def critique_report(self, topic: str, context: str, report: str) -> str:
        """Critique and improve a generated report."""
        chain = self.CRITIQUE_TEMPLATE | self.llm | self.parser
        return chain.invoke({"topic": topic, "context": context, "report": report})

