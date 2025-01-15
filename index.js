const deepspeech = require("deepspeech");
const fs = require("fs");
const path = require("path");
const ffmpeg = require("fluent-ffmpeg");
const ffmpegPath = require("@ffmpeg-installer/ffmpeg").path;

// Configure ffmpeg path
ffmpeg.setFfmpegPath(ffmpegPath);

// Load model and scorer paths
const modelPath = path.resolve("./deepspeech-0.9.3-models.pbmm");
const scorerPath = path.resolve("./deepspeech-0.9.3-models.scorer");

// Check model files
if (!fs.existsSync(modelPath) || !fs.existsSync(scorerPath)) {
  console.error("Model or scorer files not found");
  process.exit(1);
}

// Initialize DeepSpeech model
let model;
try {
  model = new deepspeech.Model(modelPath);
  model.enableExternalScorer(scorerPath);
} catch (error) {
  console.error("Error initializing DeepSpeech model:", error);
  process.exit(1);
}

async function transcribeAudio(audioFilePath) {
  if (!fs.existsSync(audioFilePath)) {
    console.error(`Audio file not found at ${audioFilePath}`);
    return;
  }

  try {
    console.log("Starting transcription...");

    // Create a stream for real-time processing
    const chunks = [];
    let currentBuffer = Buffer.alloc(0);
    const CHUNK_SIZE = 16000; // Process 1 second of audio at a time (16kHz)

    await new Promise((resolve, reject) => {
      ffmpeg(audioFilePath)
        .toFormat("wav")
        .audioChannels(1)
        .audioFrequency(16000)
        .audioCodec("pcm_s16le")
        .on("error", reject)
        .on("end", () => {
          // Process any remaining audio
          if (currentBuffer.length > 0) {
            const finalText = model.stt(currentBuffer);
            console.log(`Final chunk: "${finalText}"`);
          }
          resolve();
        })
        .pipe()
        .on("data", (chunk) => {
          // Accumulate chunks
          currentBuffer = Buffer.concat([currentBuffer, chunk]);

          // Process chunks when we have enough data
          while (currentBuffer.length >= CHUNK_SIZE) {
            const chunkToProcess = currentBuffer.slice(0, CHUNK_SIZE);
            currentBuffer = currentBuffer.slice(CHUNK_SIZE);

            // Process the chunk
            try {
              const intermediateText = model.stt(chunkToProcess);
              if (intermediateText.trim()) {
                console.log(`Intermediate result: "${intermediateText}"`);
              }
            } catch (err) {
              console.log("Error processing chunk:", err.message);
            }
          }

          chunks.push(chunk);
        });
    });

    // Process entire audio for final result
    const completeBuffer = Buffer.concat(chunks);
    const finalResult = model.stt(completeBuffer);
    console.log("\nComplete transcription:");
    console.log(finalResult);
    return finalResult;
  } catch (error) {
    console.error("Error during transcription:", error);
    throw error;
  }
}

// Example usage
const audioFilePath = path.join(__dirname, "harvard.wav");
transcribeAudio(audioFilePath)
  .then(() => console.log("Transcription process completed"))
  .catch((error) => console.error("Transcription failed:", error));
