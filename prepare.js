import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
// import { OpenAIEmbeddings } from "@langchain/openai";
import { PineconeStore } from "@langchain/pinecone";
import { Pinecone as PineconeClient } from "@pinecone-database/pinecone";

const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "gemini-embedding-001",
  outputDimensionality: 1536,
});

// const embeddings = new OpenAIEmbeddings({ model: "text-embedding-3-small" });

const pinecone = new PineconeClient();

// ✅ CORRECT
const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
  pineconeIndex,
  maxConcurrency: 5, // Default value if omitted is 10
});

export async function indexTheDocument(filePath) {
  const loader = new PDFLoader(filePath, { splitPages: false });
  const doc = await loader.load();

//   console.log(doc[0].pageContent);

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  const texts = await textSplitter.splitText(doc[0].pageContent);

  const documents = texts.map((chunk) => {
    return {
      pageContent: chunk,
      metadata: doc[0].metadata,
    };
  });

  console.log(`Documents prepared for embedding: ${documents.length} chunks`);
  const generatedEmbeddings = await vectorStore.addDocuments(documents);
  console.log("Embeddings generated and stored in Pinecone:", generatedEmbeddings);
}
