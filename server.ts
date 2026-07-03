import express from "express";
import path from "path";
import dotenv from "dotenv";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json());

// Lazy-initialize Gemini API client on the server side
let aiClient: GoogleGenAI | null = null;

function getGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("Gemini API-Key ist nicht konfiguriert. Bitte konfigurieren Sie GEMINI_API_KEY.");
  }
  if (!aiClient) {
    aiClient = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return aiClient;
}

async function generateContentWithFallback(
  ai: GoogleGenAI,
  options: {
    contents: any;
    config?: any;
  }
) {
  // Use a round-robin fallback list: gemini-3.5-flash -> gemini-3.1-flash-lite -> gemini-flash-latest
  const models = ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-flash-latest"];
  let lastError = null;
  const maxRounds = 2;

  for (let round = 1; round <= maxRounds; round++) {
    for (const model of models) {
      try {
        console.log(`[Round ${round}] Attempting generation with model: ${model}`);
        const response = await ai.models.generateContent({
          model,
          contents: options.contents,
          config: options.config,
        });
        return response;
      } catch (err: any) {
        lastError = err;
        const status = err?.status || err?.statusCode || 0;
        const message = String(err?.message || "");

        const isTransient =
          status === 503 ||
          status === 429 ||
          status === 500 ||
          message.includes("503") ||
          message.includes("500") ||
          message.includes("429") ||
          message.includes("UNAVAILABLE") ||
          message.includes("high demand") ||
          message.includes("exhausted") ||
          message.includes("temporary");

        if (isTransient) {
          console.warn(
            `Transient error with ${model} in round ${round} (status: ${status}): ${message}. Trying next available model/round...`
          );
        } else {
          // Non-transient error (e.g. invalid arguments, key auth failure) - throw immediately
          console.error(`Non-transient error with ${model}: ${message}`);
          throw err;
        }
      }
    }
    // Delay slightly between retry rounds if all models failed in the current round
    if (round < maxRounds) {
      const waitTime = 1000 * round;
      console.warn(`All models failed in round ${round}. Waiting ${waitTime}ms before the next round...`);
      await new Promise((resolve) => setTimeout(resolve, waitTime));
    }
  }
  throw lastError || new Error("Failed to generate content after trying all fallback models.");
}

// Endpoint for dynamic question/advice from the German linguistic expert
app.post("/api/chat", async (req, res) => {
  try {
    const { message, contextText, history } = req.body;
    let ai;
    try {
      ai = getGeminiClient();
    } catch (apiErr: any) {
      return res.status(500).json({
        error: "Gemini API Key is not configured in environment variables. Please configure GEMINI_API_KEY.",
      });
    }

    const systemInstruction = `Du bist ein hochqualifizierter linguistischer Assistent, spezialisiert auf deutsche Etymologie, Soziolinguistik, Sprachkontakt und historische Linguistik.
Deine Aufgabe ist es, Fragen bezüglich Entlehnungen (Lehnwörter vs. Fremdwörter), Sprachgeschichtlichen Hintergründen, etymologischen Querverbindungen und der Integration deutscher Wörter sachlich zu klären.

Beachte stets folgende wissenschaftliche Richtlinien:
1. Trenne Beobachtung (Phonologie, Orthographie, Morphologie, Syntax) und Interpretation (soziolinguistische Motive, historischer Kontext, Verdrängungsaspekte) absolut strikt voneinander!
2. Weise ausdrücklich und transparent auf wissenschaftliche Unsicherheiten oder umstrittene etymologische Theorien hin.
3. Belege deine Aussagen mit konkreten Wortbeispielen aus dem Text.
4. Beziehe deine Angaben aus linguistischen Referenzdaten und zitiere und verlinke explizit aus dem Etymologischen Wörterbuch des Deutschen (DWDS) unter https://www.dwds.de/d/wb-etymwb und dem Wörterbuchnetz unter https://woerterbuchnetz.de, um deine Antworten wissenschaftlich zu untermauern.
5. Sofern sich deine Erklärungen auf den aktuell betrachteten Text beziehen, arbeite präzise nur mit diesem vorgegebenen Text.
6. Führe Beispiele für modernen Slang oder Jargon nur dann auf, wenn diese eine nachgewiesene fremdsprachliche Herkunft (z. B. Anglizismen) besitzen.
7. Antworte auf Deutsch und pflege einen sachlichen, akademischen, aber gut verständlichen und präzisen Ton.

Der aktuell betrachtete Text lautet:
"${contextText}"`;

    const contents = [];
    if (history && Array.isArray(history)) {
      for (const turn of history) {
        contents.push({
          role: turn.role,
          parts: [{ text: turn.text }],
        });
      }
    }
    contents.push({
      role: "user",
      parts: [{ text: message }],
    });

    const response = await generateContentWithFallback(ai, {
      contents,
      config: {
        systemInstruction,
        temperature: 0.2,
      },
    });

    res.json({ text: response.text });
  } catch (error: any) {
    console.error("Error in /api/chat:", error);
    res.status(500).json({ error: error.message || "Fehler bei der Anfrage an Gemini." });
  }
});

// Endpoint for analyzing custom German texts
app.post("/api/analyze", async (req, res) => {
  try {
    const { text } = req.body;
    if (!text || typeof text !== "string") {
      return res.status(400).json({ error: "Kein Text zum Analysieren bereitgestellt." });
    }

    let ai;
    try {
      ai = getGeminiClient();
    } catch (apiErr: any) {
      return res.status(500).json({
        error: "Gemini API Key ist nicht konfiguriert. Bitte fügen Sie GEMINI_API_KEY im Secrets-Panel hinzu.",
      });
    }

    // JSON response structure to get scientific etymological annotations
    const responseSchema = {
      type: Type.OBJECT,
      properties: {
        words: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              word: { type: Type.STRING, description: "Das genaue gefundene Wort aus dem Text." },
              lemma: { type: Type.STRING, description: "Die Grundform (Nennform) des Wortes." },
              origin: { type: Type.STRING, description: "Herkunftssprache. Verwende prägnante Namen wie 'Englisch', 'Französisch', 'Latein', 'Griechisch', 'Arabisch', 'Italienisch', 'Spanisch', 'Rotwelsch', 'Tschechisch', 'Russisch', 'Niederländisch', 'Jiddisch', 'Keltisch/Andere', 'Gemischt (Mischform)' etc., basierend auf DWDS." },
              classification: { 
                type: Type.STRING, 
                description: "Einstufung des Wortes. Verwende exakt einen dieser Werte: 'Fremdwort', 'Lehnwort', 'Vollständig integriert' oder 'Unsicher'." 
              },
              originDetails: { type: Type.STRING, description: "Etymologischer Ursprung und historischer Verlauf (z.B. von lat. x über altfranzösisch y)." },
              reason: { type: Type.STRING, description: "Linguistische Begründung der Klassifizierung (warum Lehnwort oder Fremdwort based on Orthographie, Phonologie, Flexion)." },
              observation: { type: Type.STRING, description: "Objektive linguistische Beobachtung (Schreibweise, Lautung, grammatikalische Anpassung)." },
              interpretation: { type: Type.STRING, description: "Linguistische Interpretation (soziolinguistischer Status, Entlehnungsursache, historischer Sprachkontakt)." },
              reference: { type: Type.STRING, description: "Konkrete linguistische Referenzdaten mit Nennung von seriösen Online-Quellen (z.B. Duden Herkunftswörterbuch duden.de, DWDS dwds.de, Kluge)." },
              lexemeForms: { type: Type.STRING, description: "Liste oder Erklärung anderer Formen, in denen dieses Lexem im Deutschen auftreten kann (z.B. als Verb, Adjektiv und Nomen, etwa: 'Nomen: Design, Verb: designen, Adjektiv: designt'). Falls keine weiteren Schichten existieren, beschreibe die grammatische Ausprägung der Wortart." }
            },
            required: ["word", "lemma", "origin", "classification", "originDetails", "reason", "observation", "interpretation", "reference", "lexemeForms"]
          }
        }
      },
      required: ["words"]
    };

    const prompt = `Analysiere den folgenden deutschen Text linguistisch auf Entlehnungen (Wörter fremder Herkunft, insbesondere aus dem Englischen, Französischen, Italienischen, Spanischen, aber auch Latein, Griechisch, keltische Toponyme oder andere Sprachen).
Kennzeichne auch Wörter, die so stark integriert sind, dass sie im Alltagsgebrauch oft nicht mehr als Fremd-/Lehnwort wahrgenommen werden (z.B. Mode, aktuell, Galerie) und setze deren classification auf 'Vollständig integriert'.
Kennzeichne etymologisch unsichere Fälle oder Grenzgänger mit 'Unsicher'.

Wissenschaftliche Pflichten:
1. Trenne objektiv-beschreibende Beobachtung ('observation') absolut strikt von interpretierender Erklärung ('interpretation') für jedes Wort.
2. Gib konkrete linguistische Beobachtungen zu Flexion, Orthographie, Phonologie und Morphologie an.
3. Markiere für jedes erfasste Wort auch verwandte Lexeme in unterschiedlichen Wortformen im Deutschen (z. B. als Verb, Adjektiv und Nomen, und trage diese ins Feld 'lexemeForms' ein, z.B. Nomen: Post, Verb: posten, Adjektiv: gepostet).
4. Untermauere deine Klassifizierung und Angaben unbedingt mit etymologischen Belegen aus dem DWDS unter https://www.dwds.de/d/wb-etymwb und dem Wörterbuchnetz unter https://woerterbuchnetz.de im Feld 'reference'.
5. Arbeite bei der Textanalyse AUSSCHLIESSLICH mit dem hier vorgegebenen Text. Extrahiere keine Wörter, die nicht im Text enthalten sind.
6. Slang- oder Jargonbegriffe sind nur aufzuführen, wenn sie eine nachweisbare fremdsprachige Herkunft besitzen.

Text:
"${text}"`;

    const response = await generateContentWithFallback(ai, {
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema,
        temperature: 0.1,
      },
    });

    res.json(JSON.parse(response.text || "{}"));
  } catch (error: any) {
    console.error("Error in /api/analyze:", error);
    res.status(500).json({ error: error.message || "Fehler bei der linguistischen Analyse." });
  }
});

// Setup Vite Dev Server / Static Middleware
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    console.log("Starting server in DEVELOPMENT mode with Vite Middleware...");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    console.log("Starting server in PRODUCTION mode...");
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Linguistic backend server listening on http://localhost:${PORT}`);
  });
}

startServer();
