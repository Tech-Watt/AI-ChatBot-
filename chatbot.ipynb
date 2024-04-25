{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "from dotenv import load_dotenv\n",
    "import os \n",
    "from langchain_community.llms import Ollama\n",
    "from langchain_openai.chat_models import ChatOpenAI\n",
    "from langchain.prompts import PromptTemplate\n",
    "from langchain_core.output_parsers import StrOutputParser\n",
    "from langchain.text_splitter import RecursiveCharacterTextSplitter\n",
    "from langchain_community.document_loaders import TextLoader\n",
    "from langchain_openai.embeddings import OpenAIEmbeddings\n",
    "from langchain_community.vectorstores import FAISS\n",
    "from langchain_core.runnables import RunnablePassthrough, RunnableParallel"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "ollama_llm = Ollama(model = 'llama3')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "load_dotenv()\n",
    "API_KEY = os.getenv('OPENAI_API_KEY')\n",
    "Model = 'gpt-3.5-turbo'\n",
    "gpt_llm = ChatOpenAI(api_key = API_KEY,model= Model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "AIMessage(content='A bot, short for robot, is a software program that performs automated tasks on the internet. Bots can be programmed to perform a variety of functions, such as answering questions, providing information, or interacting with users in a chat or messaging platform. Bots can be simple or complex, and can be used for a wide range of purposes, including customer service, marketing, and entertainment.', response_metadata={'token_usage': {'completion_tokens': 78, 'prompt_tokens': 11, 'total_tokens': 89}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_c2295e73ad', 'finish_reason': 'stop', 'logprobs': None}, id='run-22379d8e-e934-4422-8544-2200c1b90c5b-0')"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "gpt_llm.invoke('what is a bot')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'A bot, short for robot, is a program or software application that performs automated tasks on the internet. Bots can be designed to perform a wide range of functions, from simple tasks like web scraping and data collection to more complex tasks like online customer service and chatbot interactions. Bots can be used for both beneficial and malicious purposes, so it is important to be cautious when interacting with them online.'"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "parser = StrOutputParser()\n",
    "gpt_chain = gpt_llm|parser\n",
    "gpt_chain.invoke('what is a bot')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "loader = TextLoader('data.txt',encoding = 'utf-8')\n",
    "document = loader.load()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Document(page_content='first.  do I create a zap first ?\",\"completion\":\" First you create your template video, and then you connect it to Zapier.  https://www.bhuman.ai/tutorials  Here are some step by step walkthroughs', metadata={'source': 'data.txt'})"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "spliter = RecursiveCharacterTextSplitter(chunk_size = 200,chunk_overlap = 50)\n",
    "chunks = spliter.split_documents(document)\n",
    "chunks[3]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "vector_storage = FAISS.from_documents(chunks,OpenAIEmbeddings())\n",
    "retriever = vector_storage.as_retriever()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[Document(page_content='BHuman is an incredible tool for engagement, customer loyalty, differentiation, and overall success. Please click on the link below to read more about BHuman:', metadata={'source': 'data.txt'}),\n",
       " Document(page_content=\"can only be cloned with the paid plan. However, you'll have 7 days free trial and may cancel you subscription any time. More info about our pricing can be found here:  https://www.bhuman.ai/pricing\", metadata={'source': 'data.txt'}),\n",
       " Document(page_content='our enterprise white glove services here and book a call with the team  https://www.bhuman.ai/pricing\"}, {\"prompt\":\" how do I get the lifetime deal available on appsumo?\",\"completion\":\" BHuman', metadata={'source': 'data.txt'}),\n",
       " Document(page_content='with your audience at scale, whether it be leads, customers, or employees. Please click on the link below to read more about BHuman: https://docs.bhuman.ai/en/articles/6942071-what-is-bhuman\"},', metadata={'source': 'data.txt'})]"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "retriever.invoke('what is the pricing for Bhuman')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "template = (\"\"\"\n",
    "You are AI-powered chatbot designed to provide \n",
    "information and assistance for customers\n",
    "based on the context provided to you only. \n",
    "            \n",
    "Context:{context}\n",
    "Question:{question}\n",
    "\"\"\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'\\nYou are AI-powered chatbot designed to provide \\ninformation and assistance for customers\\nbased on the context provided to you only. \\n            \\nContext: Here is a context to use\\nQuestion: This is a question to answer\\n'"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "prompt = PromptTemplate.from_template(template=template)\n",
    "prompt.format(\n",
    "    context = ' Here is a context to use',\n",
    "    question = ' This is a question to answer'\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "result = RunnableParallel(context = retriever,question = RunnablePassthrough())\n",
    "chain = result |prompt | gpt_llm | parser"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'The pricing plan for BHuman includes a paid plan which can only be cloned with the paid plan. However, there is a 7-day free trial available and you can cancel your subscription at any time. More information about the pricing can be found on their website at https://www.bhuman.ai/pricing. Additionally, there are different plans available such as the GROWTH plan for $39/month which is recommended for startups and includes 200 videos a month.'"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "chain.invoke('What is the pricing plan for Bhuman')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'BHuman is a platform that allows you to personalize videos at scale by cloning the face and voice of the person in the video. It is an incredible tool for engagement, customer loyalty, differentiation, and overall success. You can read more about BHuman by clicking on the link provided: https://docs.bhuman.ai/en/articles/6942071-what-is-bhuman'"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "chain.invoke('What is Bhuman')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'Based on the context provided, it seems like you may be experiencing issues with the clarity of your videos. One suggestion mentioned in the documents is to adjust the lighting for better visibility. Try using low lighting instead of bright, well-lit lighting to make it easier to see your mouth and improve the clarity of your videos. Additionally, ensure that you are following the tutorial carefully to generate better quality videos. If you need more detailed suggestions, feel free to reach out for further assistance.'"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "chain.invoke('My video is not clear')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": ".venv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
