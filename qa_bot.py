from langchain_community.llms import CTransformers
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS

model_file = "model/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "vectorstores/db_faiss"

def load_llm(model_file):
    llm = CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.01
    )
    return llm

def create_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt

def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False,
        chain_type_kwargs={'prompt': prompt}
    )
    return llm_chain 

def read_vectors_db():
    embedding_model = GPT4AllEmbeddings(model_file="model/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    return db

db = read_vectors_db()
llm = load_llm(model_file)

template = """<[im_start]>system sử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo câu trả lời saisai
{context}<[im_end]>\n<[im_start]>user\n{question}<[im_end]>\n<[im_start]>assistant"""

prompt = create_prompt(template)

llm_chain = create_qa_chain(prompt, llm, db)

question = "Vua Lạc Long Quân lấy nàng Âu Cơ sinh ra bao nhiêu con trai?"
response = llm_chain.invoke({"query": question})
print(response)