// creates MCP server
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { ChromaClient } from "chromadb";
import { DefaultEmbeddingFunction } from "@chroma-core/default-embed";
import {
  ListToolsRequestSchema,
  CallToolRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

// lets AI communicate with server
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

// read folders, files
import fs from "fs/promises";
import { type } from "os";

import path, { resolve } from "path";
import { spawn } from "child_process";
import { text } from "stream/consumers";
import fsSync, { stat } from "fs";

let knowledgeBase = [];
const supportedFormats = [".txt", ".md", ".pdf", ".docx"];

// const currentDir = path
//   .dirname(new URL(import.meta.url).pathname)
//   .replace(/^\/([A-Z]:)/, "$1");
//const embedScriptPath = path.join(currentDir, "embed_server.py");
//const cvadm_path = path.join(currentDir, "Conversion-Admin");

const parserScriptPath =
  "D:/work_dsi/My Learnings/Projects/ConversionAdmin-RAG-MCP/docling_parser.py";

const parserProcess = spawn("python", [parserScriptPath]);
let parserResolvers = [];
let parserBuffer = "";

parserProcess.stdout.on("data", (data) => {
  parserBuffer += data.toString();

  let lines = parserBuffer.split("\n");

  parserBuffer = lines.pop();

  for (const line of lines) {
    if (!line.trim()) continue;

    const resolve = parserResolvers.shift(); // removes first item and returns, first item means the oldest data being embeded has arrived

    if (resolve) {
      resolve(JSON.parse(line));
    }
  }
});

parserProcess.stderr.on("data", (data) => {
  console.error("PYTHON ERROR:", data.toString());
});

parserProcess.on("close", (code) => {
  console.error("Parser process closed with code", code);
});

const embedScriptPath =
  "D:/work_dsi/My Learnings/Projects/ConversionAdmin-RAG-MCP/embed_server.py";
const cvadm_path =
  "D:/work_dsi/My Learnings/Projects/ConversionAdmin-RAG-MCP/Conversion-Admin";

const embedProcess = spawn("python", [embedScriptPath]);
let pendingResolvers = [];

let buffer = "";
// listener, called everytime data has arrived after python writes to stdout
embedProcess.stdout.on("data", (data) => {
  buffer += data.toString();

  let lines = buffer.split("\n");

  buffer = lines.pop();

  for (const line of lines) {
    if (!line.trim()) continue;

    const pending = pendingResolvers.shift(); // removes first item and returns, first item means the oldest data being embeded has arrived

    if (pending) {
      const response = JSON.parse(line);

      if (response.success) {
        pending.resolve(response.embedding);
      } else {
        pending.reject(new Error(response.error));
      }
    }
  }
});

embedProcess.on("close", (code) => {
  console.error("Python process closed with code", code);
});

const server = new Server(
  {
    name: "conversionadmin-server",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  },
);

// Initialize chroma
console.error("Connecting to ChromaDB...");
const client = new ChromaClient();
console.error("Creating collection...");
const collection = await client.getOrCreateCollection({
  name: "conversion-admin-docs",
  //embeddingFunction: new DefaultEmbeddingFunction(), // much slower compared to persistent python embedding
});
console.error("Chroma connected successfully");

// document parsing using Docling
async function parserDocument(filePath) {
  return new Promise((resolve) => {
    parserResolvers.push(resolve);
    parserProcess.stdin.write(filePath + "\n");
  });
}

// gets raw paths of all files in the directory
async function getAllFiles(directoryPath) {
  let files = [];

  const items = await fs.readdir(directoryPath, { withFileTypes: true });

  for (const item of items) {
    const fullPath = path.join(directoryPath, item.name);

    // if item is a folder
    if (item.isDirectory()) {
      // recursively get all files inside it
      const nestedFiles = await getAllFiles(fullPath);
      files = files.concat(nestedFiles);
    } else {
      files.push(fullPath);
    }
  }

  return files;
}

function normalizeText(value) {
  if (typeof value === "string") {
    return value.trim();
  }

  if (Array.isArray(value)) {
    return value.join(" ").trim();
  }

  if (value == null) {
    return "";
  }

  return String(value).trim();
}

// creates vectors from the chunks using all-MiniLM-L6-v2
async function createEmbedding(text) {
  if (!text?.trim()) {
    throw new Error("Empty embedding text");
  }
  console.error(`creating embedding for: ${text}`);
  return new Promise((resolve, reject) => {
    pendingResolvers.push({ resolve, reject });
    embedProcess.stdin.write(JSON.stringify({ text }) + "\n");
  });
}

// [NOT USED] calculates the cosine similarity (angle) between two vectors, 1 is similar, 0 is unrelated
function cosineSimilarity(vecA, vecB) {
  let dotProduct = 0;
  let magnitudeA = 0;
  let magnitudeB = 0;

  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    magnitudeA += vecA[i] * vecA[i];
    magnitudeB += vecB[i] * vecB[i];
  }

  magnitudeA = Math.sqrt(magnitudeA);
  magnitudeB = Math.sqrt(magnitudeB);

  if (magnitudeA === 0 || magnitudeB === 0) return 0;

  return dotProduct / (magnitudeA * magnitudeB);
}

function buildSemanticChunks(parsedBlocks) {
  console.error("buildSemanticChunks enter...");
  const chunks = [];
  let currentHeading = "";

  for (const block of parsedBlocks) {
    if (block.label.includes("header")) {
      currentHeading = block.text;
      continue;
    }

    chunks.push({
      text: `${currentHeading}\n${block.text}`,
      heading: currentHeading,
    });
  }

  return chunks;
}

// form chunks from the text
function chunkText(text, chunkSize = 800, overlap = 150) {
  const sentences = text.match(/[^.!?\n]+[.!?\n]+/g) || [text];

  const chunks = [];

  let currentChunk = "";

  for (const sentence of sentences) {
    if (currentChunk.length + sentence.length > chunkSize) {
      chunks.push(currentChunk.trim());

      currentChunk = currentChunk.slice(-overlap) + sentence;
    } else {
      currentChunk += sentence;
    }
  }

  if (currentChunk.trim()) {
    chunks.push(currentChunk.trim());
  }

  return chunks;
}

// index only newly added, updated, deleted data, based on hash
async function incrementalIndex() {
  console.error("Starting Incremental knowdledge base build");
  const allFiles = await getAllFiles(cvadm_path);

  for (const file of allFiles) {
    if (!supportedFormats.includes(path.extname(file).toLowerCase())) continue;

    const stats = await fs.stat(file);

    const existing = await collection.get({
      where: {
        file: file,
      },
    });

    // file already indexed
    if (existing.metadatas?.length > 0) {
      const indexModified = existing.metadatas[0].lastModified;

      // unchanged file
      if (indexModified === stats.mtimeMs) {
        console.error(`Skipping unchanged file : ${file}`);
        continue;
      }

      // remove old file data
      await collection.delete({
        where: {
          file: file,
        },
      });
      // file exists, but original file has been modified
      console.error(`Re-indexing modified file: ${file}`);
    } else console.error(`indexing new file: ${file}`);

    // read new / updated file
    const parsed = await parserDocument(file);

    const chunks = buildSemanticChunks(parsed);

    const ids = [];
    const documents = [];
    const embeddings = [];
    const metadatas = [];

    for (let index = 0; index < chunks.length; index++) {
      const chunk = chunks[index];
      console.error(chunk + "\n");
      if (
        !chunk.text ||
        typeof chunk.text !== "string" ||
        !chunk.text?.trim()
      ) {
        console.error(`Invalid chunk: ${chunk}`);
        continue;
      }
      console.error(`Embedding chunk ${index + 1}/${chunks.length}` + "\n");
      const cleanText = normalizeText(chunk.text);
      if (!cleanText) continue;
      try {
        const embedding = await createEmbedding(cleanText);

        ids.push(`${file}-${index}`);
        documents.push(chunk.text);
        if (
          !Array.isArray(embedding) ||
          !embedding.every((n) => typeof n === "number")
        ) {
          console.error(`Invalid embedding at chunk ${index}`);
          continue;
        }
        embeddings.push(embedding);
        metadatas.push({
          file,
          chunkIndex: index,
          source: "conversion-admin",
          lastModified: stats.mtimeMs,
          heading: chunk.heading,
        });
      } catch (error) {
        console.error(`Failed embedding chunk ${index} in ${file}`);
        console.error(error.message);
        continue;
      }
    }
    await collection.add({
      ids,
      documents,
      embeddings,
      metadatas,
    });
    console.error(`Finished indexing: ${file}`);
  }
}

// index data and store in chromadb
async function buildKnowledgeBase() {
  console.error("Building knowledge base...");

  const allFiles = await getAllFiles(cvadm_path);

  for (const file of allFiles) {
    if (!supportedFormats.includes(path.extname(file).toLowerCase())) continue;
    console.error(`Indexing : ${file}`);
    const stats = await fs.stat(file);
    const parsed = await parserDocument(file);
    const chunks = buildSemanticChunks(parsed);
    const ids = [];
    const documents = [];
    const embeddings = [];
    const metadatas = [];

    for (let index = 0; index < chunks.length; index++) {
      console.error(`Embedding chunk ${index + 1}/${chunks.length}`);

      const chunk = chunks[index];
      console.error(chunk.text + "\n");
      if (
        !chunk.text ||
        typeof chunk.text !== "string" ||
        !chunk.text?.trim()
      ) {
        console.error(`Invalid chunk: ${chunk}`);
        continue;
      }
      const cleanText = normalizeText(chunk.text);
      if (!cleanText) continue;
      try {
        // local persistent MiniLM
        const embedding = await createEmbedding(cleanText);

        ids.push(`${file}-${index}`);

        documents.push(chunk.text);

        if (
          !Array.isArray(embedding) ||
          !embedding.every((n) => typeof n === "number")
        ) {
          console.error(`Invalid embedding at chunk ${index}`);
          continue;
        }
        embeddings.push(embedding);

        metadatas.push({
          file,
          chunkIndex: index,
          source: "conversion-admin",
          lastModified: stats.mtimeMs,
          heading: chunk.heading,
        });
      } catch (error) {
        console.error(`Failed embedding chunk ${index} in ${file}`);
        console.error(error.message);
        continue;
      }
    }

    // Store everything in Chroma
    await collection.add({
      ids,
      documents,
      embeddings,
      metadatas,
    });

    console.error(`Indexed ${file}`);
  }

  console.error("Knowledge base built successfully");
}

// performs semantic search in db using user query
async function semanticSearch(query) {
  console.error("semanticsearch begins");
  // chromadb embed using the same miniLM model internally
  const queryEmbedded = await createEmbedding(query);
  let results = await collection.query({
    queryEmbeddings: [queryEmbedded],
    nResults: 10,
  });
  console.error(results.documents[0]);

  return results;
}

// Tells AI which tools exist
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "search_cvadmdocs",

        description:
          "Semantic search over Conversion Admin / Derived Output / DO documentation, monitor conversion jobs, APIs, web services, events, business rules, workflows, TXT files, and internal technical docs. Use this tool whenever the user asks anything about Conversion Admin functionality, jobs, APIs, integrations, events, rules, architecture, or implementation details.",

        inputSchema: {
          type: "object",

          properties: {
            query: {
              type: "string",

              description: "Search query",
            },
          },

          required: ["query"],
        },
      },
    ],
  };
});

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const toolName = request.params.name;
  const args = request.params.arguments;

  if (toolName === "search_cvadmdocs") {
    console.error("search_cvadmdocs tool is called...");
    const existing = await collection.count();
    if (existing === 0) {
      await buildKnowledgeBase();
    }
    const results = await semanticSearch(args.query);
    const formatted = results.documents[0]
      .map((doc, index) => {
        return `
        Result ${index + 1}

        File:
        ${results.metadatas[0][index].file}

        Content:
        ${doc}
        `;
      })
      .join("\n-----------------\n");
    return {
      content: [
        {
          type: "text",
          text: formatted,
        },
      ],
    };
  }
});

const transport = new StdioServerTransport();

await server.connect(transport);

console.error("MCP SERVER READY");

// Run indexing in background AFTER connection
setTimeout(async () => {
  try {
    const existing = await collection.count();

    if (existing === 0) {
      await buildKnowledgeBase();
    } else {
      await incrementalIndex();
    }

    console.error("Indexing complete");
  } catch (err) {
    console.error("Startup error:", err);
  }
}, 1000);

// test
//await buildKnowledgeBase();
