from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings

pdf_data_path = "data"
vector_db_path = "vectorstores/db_faiss"

def create_db_from_text():
    raw_text = """Trong lễ tưởng niệm bác sĩ Nguyễn Khắc Viện tại Paris, Tiến sĩ Sử học Charles Fourniau, Chủ tịch Hội Hữu nghị Pháp - Việt đã viết: "... Ngay từ những phút đầu tiên, tôi nhận ra ngay đây sẽ là bậc thầy của tôi. Và ông mãi vẫn là bậc thầy của tôi. Tôi may mắn được tiếp cận ông 
một trong những trí tuệ sáng chói nổi bật nhất. Vốn văn hóa của ông , hay nói đúng ra là vốn các văn hóa của ông, bởi lẽ ông có đến ba vốn văn hóa, Việt Nam, Trung Hoa, Pháp... quả thật dường như là vô hạn..." Nhiều tác phẩm của bác sĩ Nguyễn Khắc Viện cho đến nay vẫn còn
nhưng giá trị lớn lao. Bất cứ đối tượng nào, từ người già, thanh niên đến trẻ em đều có thể thấy qua tác phẩm của ông bóng dáng một người bạn , một người thầy, một người ông với kiến thức uyên thâm và tấm lòng nhân ái.
Về một số vấn đề có tính thời sự, trong dịp tái bản cuốn Một đôi lời trước ngày đi xa một năm, ông đã viết: "... Mới hơn 10 năm mà nay nhớ lại nhiều việc, như là chuyện thời xa xưa, cả nước đã chuyển sang một thời đại mới. Nay cho in lại, xin cứ giữ nguyên bản, không sửa chữa, như là một vết tích của một thời, để bạn đọc ngày nay thấy một số người "xưa kia" suy nghĩ những gì... Thời thế thay đổi, không thể không thay đổi ý kiến, loại trừ một số sai lầm tư tưởng,nhưng điều không thể thay đổi là cái đạo lý làm người. Thức thời, chứ không phải cơ hội..." Quả là toàn bộ tác phẩm của Nguyễn Khắc Viện, kể cả những đề tài "thời sự" đã qua như phong trào "hợp tác xã" hay " Liên Xô" ... vẫn sáng rõ một
"đạo lý" đẹp đẽ và chung thủy của một sĩ phu trung thực, hết lòng vì nước vì dân, nên đều có giá trị bổ sung kiến thức, bồi dưỡng tâm hồn cho nhiều thế hệ bạn đọc.
Với sự ngưỡng mộ và kính trọng đặc biệt sâu sắc với bác sĩ Nguyễn Khắc Viện, được sự đồng ý và cộng tác của gia đình cố bác sĩ Nguyễn Khắc Viện, Công ty Cổ phần Sách Thái Hà xin giới thiệu với độc giả bộ sách gồm 5 cuốn: Tâm tình của đất nước, Đạo và Đời, Việt Nam - Một thiên lịch sử, Nguyễn Khắc Viện - Chân dung và kỷ niệm, Tự truyện.
Trân trọng giới thiệu cùng độc giả và rất mong nhận được ý kiến đóng góp của bạn đọc gần xa."""
    
    text_splitter = CharacterTextSplitter(
        separator="/n",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len

    )
    
    chunks = text_splitter.split_text(raw_text)

    embedding_model = GPT4AllEmbeddings(model_file = "model/all-MiniLM-L6-v2-f16.gguf")

    db = FAISS.from_texts(texts=chunks, embedding=embedding_model)
    db.save_local(vector_db_path)

    return db

def create_db_from_files():
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    embedding_model = GPT4AllEmbeddings(model_file="model/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)
    return db

create_db_from_files()



    


