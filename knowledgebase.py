from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.vectordb.pgvector import PgVector
from agno.embedder.google import GeminiEmbedder

pdf_knowledge_base = PDFKnowledgeBase(
    path="cvs",
    vector_db=PgVector(
        table_name="cvs",
        embedder=GeminiEmbedder(dimensions=1536),
        db_url="postgresql://neondb_owner:npg_ewUNJjB7Qn8T@ep-muddy-mouse-a5a04l1u-pooler.us-east-2.aws.neon.tech/hackothon?sslmode=require",
    ),
   reader=PDFReader()
)

# jd_knowledge_base = PDFKnowledgeBase(
#     path="cvs/jd1.pdf",
#     vector_db=PgVector(
#         table_name="jd",
#         embedder=GeminiEmbedder(dimensions=1536),
#         db_url="postgresql://neondb_owner:npg_ewUNJjB7Qn8T@ep-muddy-mouse-a5a04l1u-pooler.us-east-2.aws.neon.tech/hackothon?sslmode=require",
#     ),
#    reader=PDFReader()
# )