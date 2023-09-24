import streamlit as st
from langchain.agents import load_tools, AgentType
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import os
import pprint
from langchain.utilities import GoogleSerperAPIWrapper
import wikipedia
import pandas as pd

import requests
from bs4 import BeautifulSoup


def google_search_urls(keywords,urls):
    search_google = GoogleSerperAPIWrapper()
    results = search_google.results(keywords)
    for result in results['organic']:
        urls.append(result['link'])
    return urls




  

def wikipedia_search_urls(keywords,urls):
    keyword_list = [keyword.strip() for keyword in keywords.split(',')]
    for keyword in keyword_list:
        try:
            results = wikipedia.search(keyword)
            article = wikipedia.page(results[0])
            urls.append(article.url)
        except wikipedia.exceptions.DisambiguationError as e:
            print(str(e.options))
        except wikipedia.exceptions.PageError as e:
            print(f"PageError: {e}")    
    return urls

def keywords_e(question):
    prompt_keyword = PromptTemplate(
    input_variables=["text"],
    template="Please extract keywords for the following question, using Spaces instead of commas between keywords,question:{text}?"
    )
    llm = OpenAI(model_name="gpt-3.5-turbo")
    keywords=llm(prompt_keyword.format(text=question))
    return keywords

# Streamlit界面布局和逻辑
def main():
    os.environ["SERPER_API_KEY"] = "your google serper api key"
    serper_api_key = os.environ["SERPER_API_KEY"]
    os.environ["OPENAI_API_KEY"] = "your openai api key"
    openai_api_key = os.environ["OPENAI_API_KEY"]
    st.title("Search GPT")
    llm = OpenAI(model_name="gpt-3.5-turbo")

    # 输入问题
    question = st.text_input("Question：")
    

    # 搜索并显示结果
    if question:
        keywords=keywords_e(question)

        urls=[]
        google_search_urls(keywords,urls)
        wikipedia_search_urls(keywords,urls)
        url_data = pd.DataFrame(columns=['url', 'summary', 'score'])
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
        summary_template = """
        Analyze the following text, give a summary of the text according to the question
        text：{context}
        question: {question}
        summary:"""

        SUMMARY_CHAIN_PROMPT = PromptTemplate.from_template(summary_template)

        score_template = """Analyze the following text and give a score based on whether it matches the question, out of 100 points, you only need to give the number of the final score, you do not need to give a reason分析以下文本，并根据是否匹配问题给出一个得分，满分为100分，你只需要给出最终得分的数字，不需要给出理由
        text：{context}
        question: {question}
        score:"""

        SCORE_CHAIN_PROMPT = PromptTemplate.from_template(score_template)


        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
            
        vectorstore={}
            
        for url in urls:
            response = requests.get(url, verify=False)
            soup = BeautifulSoup(response.content, 'html.parser')
            for ad in soup.find_all(class_='ad'):
                ad.decompose()
            text = soup.get_text()
            
            all_splits = text_splitter.split_text(text)
            
            
            #summary = llm(prompt_summary.format(question=question,context=text))
            #score = llm(prompt_score.format(question=question,context=text))
            
            vectorstore[url] = Chroma.from_texts(texts=all_splits, embedding=OpenAIEmbeddings())
            
            summary_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore[url].as_retriever(),
            chain_type_kwargs={"prompt": SUMMARY_CHAIN_PROMPT})
            
            summary = summary_chain({"query": question})["result"]
            
            score_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore[url].as_retriever(),
            chain_type_kwargs={"prompt": SCORE_CHAIN_PROMPT})
            
            score = int(score_chain({"query": question})["result"])
            
            
            data = [
            {'url': url, 'summary':summary, 'score': score}
            ]
            url_data = pd.concat([url_data, pd.DataFrame(data)])
        sorted_data=url_data.sort_values('score', ascending=False)
        st.subheader("搜索结果：")
        for index, row in sorted_data.iterrows():
            st.write(f"url：{row['url']}")
            st.write("summary:")
            st.write(row['summary'])




# 运行Streamlit应用
if __name__ == "__main__":
    main()