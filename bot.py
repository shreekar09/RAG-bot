import os
os.environ['OPENAI_API_KEY']=''

from langchain_community.document_loaders.csv_loader import CSVLoader
csv_path = r"C:\Users\KIIT\Documents\college\projects\chatbot\genbot\employee_data.csv"
loader = CSVLoader(file_path=csv_path)
document = loader.load()
print(document)

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(separators='<END>')
docs = text_splitter.split_documents(document)

# for i in docs:
#     print(i)

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
embeding_data = []
model_name = "BAAI/bge-large-en"
model_kwargs = {'device':'cpu'}
# encode_kwargs = {'normalize-embeddings': False}

hf=HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    # encode_kwargs=encode_kwargs
)
# for i in docs:
#     emb = hf.embed_query(i.page_content)
#     print(emb, '****\n\n')

from langchain_community.vectorstores import Chroma
docsearch = Chroma.from_documents(docs, hf).as_retriever()

from langchain.chains import retrieval_qa
from langchain.llms.openai import OpenAI
qa = retrieval_qa.from_chain_type(Llm = OpenAI(temperature=0.8), chain_type="stuff", retriever=docsearch)

while True:
    Query = input('Ask a Query:\t')
    output = qa.run(Query)
    print('\n\n**************************************\n', output, '\n***********')