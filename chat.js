import readline from 'node:readline/promises';
import Groq from "groq-sdk";
import { vectorStore } from './prepare.js';

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
export async function chat() {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    });
    while (true) {
        const question = await rl.question('you: ')
        if (question === '/bye') {
            break;
        }

        const relevantChunks = await vectorStore.similaritySearch(question, 5);

        const context = relevantChunks.map((chunk) => chunk.pageContent).join("\n\n");

        const SYSTEM_PROMPT = `You are a helpful assistant for company employees. Use the following context to answer the question.
If the context is not enough to answer the question, say "I don't know".

Output formatting rules:
- Format the response as plain text suitable for reading in a terminal.
- Do NOT use markdown formatting (no bold/italics with asterisks, no hash headers like ###, - **, and NO markdown tables).
- If presenting multiple items or comparisons, use simple bullet points with plain dashes (-) or numbered lists.
- Keep the response clean, readable, and well-spaced.`;

        const userQuery = `Question: ${question}
        Context: ${context}
        Answer:`
        const completion = await groq.chat.completions.create({
            messages: [
                {
                    role: "system",
                    content: SYSTEM_PROMPT,
                },
                {
                    role: "user",
                    content: userQuery,
                },
            ],
            model: "openai/gpt-oss-20b",
        });
        console.log(`\nAnswer:\n${completion.choices[0].message?.content}\n`);
    }


    rl.close();
}

chat();