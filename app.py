import streamlit as st
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.baidusearch import BaiduSearchTools
from agno.tools.crawl4ai import Crawl4aiTools
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_newsletter():
    with st.spinner('Generating your newsletter...'):
        try:
            # Create the agent
            newsletter_agent = Agent(
                name="NewsletterAgent",
                role="An expert newsletter generator for the event industry that creates engaging content about Event Tech News, Big Tech & Martech News, LinkedIn & Social Media News, and AI News.",
                instructions=[
                    "You are an expert newsletter generator for the event industry.",
                    "Your task is to create a newsletter featuring four key categories: Event Tech News, Big Tech & Martech News, LinkedIn & Social Media News, and AI News.",
                    "For each category, find at least three highly relevant and recent articles that speak directly to event professionals.",
                    "For each article, provide:",
                    "1. A short, punchy headline.",
                    "2. A concise 1-2 line summary that captures the key points.\n",
                    "3. A 'Dahlia's Take' commentary explaining why this news matters for event professionals and what actionable steps they should consider next.",
                    "Use clear emojis for each category header (e.g., 🚀 for Event Tech News), ensure the stories are directly relevant to event operations, planning, and industry trends, and maintain a conversational, no-nonsense tone with clever humor.",
                    "Avoid including stories not relevant to the event industry.",
                    "Format the newsletter with sequential numbering and bullet points where applicable.",
                    "Output the final newsletter in markdown format with proper headers and structure.",
                    "Include sources of each news summary."
                ],
                model=OpenAIChat(id="gpt-4.5-preview"),
                tools=[BaiduSearchTools(), Crawl4aiTools()],
                add_datetime_to_instructions=True,
            )
            
            # Get the response directly (no streaming for more reliable output)
            response = newsletter_agent.run(
                "Generate a newsletter featuring the latest developments in Event Tech News, Big Tech & Martech News, LinkedIn & Social Media News, and AI News, specifically tailored for event professionals."
            )
            
            # Handle different response types
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, str):
                return response
            else:
                return str(response)
                
        except Exception as e:
            st.error(f"Error generating newsletter: {str(e)}")
            return None

def main():
    st.title("Event Industry Newsletter Generator")
    st.write("""
    Generate a personalized newsletter featuring the latest developments in the event industry.
    The newsletter covers four key categories:
    - Event Tech News
    - Big Tech & Martech News
    - LinkedIn & Social Media News
    - AI News
    """)
    
    if st.button("Generate Newsletter"):
        result = generate_newsletter()
        if result:
            st.markdown("### Your Newsletter")
            # Display the result in a scrollable container
            st.markdown(result)

if __name__ == "__main__":
    main()