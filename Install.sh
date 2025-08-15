
# Build faiss customized library
cd faiss
sh build_from_source.sh
sh install.sh

# Build HedraRAG
cd ../HedraRAG
pip install -e .

# Build Langchain (Optional)
cd ../LangChain
pip install -r requirements.txt
